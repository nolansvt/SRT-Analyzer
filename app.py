import os
import json
import tempfile
import traceback
import gradio as gr
from config import config
from transcription.gladia_client import GladiaClient
from analysis.comparator import compare_srt, compare_four_way
from analysis.wer import WERResult, compute_wer
from analysis.srt_parser import srt_file_to_plain_text
from llm.factory import get_llm_client
from llm.srt_corrector import correct_srt
from utils.vocab_extractor import extract_vocabulary_from_srts, merge_vocabularies
from utils.rag_retriever import RAGRetriever

rag = RAGRetriever("data/")


def load_glossary() -> list[str]:
    path = config.resolved_vocabulary_path
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        glossary = [entry["value"] for entry in data if "value" in entry]
        print(f"[Glossaire] {len(glossary)} termes chargés")
        return glossary
    print("[Glossaire] Aucun fichier trouvé")
    return []


glossary = load_glossary()


def match_files_by_name(media_files, srt_files):
    srt_map = {os.path.splitext(os.path.basename(f.name))[0]: f for f in srt_files}
    pairs, unmatched = [], []
    for media in media_files:
        stem = os.path.splitext(os.path.basename(media.name))[0]
        if stem in srt_map:
            pairs.append((media, srt_map[stem]))
        else:
            unmatched.append(stem)
    return pairs, unmatched


def srt_to_tempfile(content: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def substitutions_to_tempfile(examples: list[tuple[str, str]]) -> str:
    lines = ["Référence,Prédiction"] + [f'"{r}","{h}"' for r, h in examples]
    return srt_to_tempfile("\n".join(lines), "_substitutions.csv")


def list_to_tempfile(items: list[str], suffix: str) -> str:
    return srt_to_tempfile("\n".join(items), suffix)


def format_metrics(label: str, wer: WERResult, hypothesis_label: str = "Prédit") -> str:
    lines = [
        f"## {label}\n",
        "| Métrique | Description | Valeur |",
        "|---|---|---|",
        f"| WER | % de mots incorrects | **{wer.wer:.1%}** |",
        f"| MER | % de mots de référence mal alignés | {wer.mer:.1%} |",
        f"| WIL | % d'information perdue par mot | {wer.wil:.1%} |",
        f"| Mots référence | Mots dans le texte corrigé | {wer.ref_word_count} |",
        f"| Mots prédits | Mots dans la transcription | {wer.hyp_word_count} |",
        "\n| Type d'erreur | Description | Nombre |",
        "|---|---|---|",
        f"| Substitutions | Mots remplacés par un autre mot | {wer.substitutions} |",
        f"| Insertions | Mots ajoutés en trop | {wer.insertions} |",
        f"| Suppressions | Mots manquants | {wer.deletions} |",
    ]
    if wer.substitution_examples:
        total = len(wer.substitution_examples)
        lines.append(f"\n### Substitutions (10 / {total})\n| Référence | {hypothesis_label} |\n|---|---|")
        for r, h in wer.substitution_examples[:10]:
            lines.append(f"| {r} | {h} |")
    if wer.insertion_examples:
        total = len(wer.insertion_examples)
        lines.append(f"\n### Insertions (10 / {total})")
        for w in wer.insertion_examples[:10]:
            lines.append(f"- {w}")
    if wer.deletion_examples:
        total = len(wer.deletion_examples)
        lines.append(f"\n### Suppressions (10 / {total})")
        for w in wer.deletion_examples[:10]:
            lines.append(f"- {w}")
    return "\n".join(lines)


def load_vocabulary(vocab_file=None) -> tuple[list | None, str]:
    if vocab_file is not None:
        try:
            with open(vocab_file.name, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            return vocab, f"✅ Vocabulaire : {len(vocab)} termes (fichier uploadé)"
        except Exception as e:
            return None, f"⚠️ Erreur lecture vocabulaire : {e}"

    path = config.resolved_vocabulary_path
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            return vocab, f"✅ Vocabulaire : {len(vocab)} termes ({os.path.basename(path)})"
        except Exception as e:
            return None, f"⚠️ Erreur lecture vocabulaire : {e}"

    return None, "ℹ️ Aucun vocabulaire — Gladia CV = Gladia base"


def refresh_vocab_status(vocab_file):
    _, status = load_vocabulary(vocab_file)
    return status


def analyze_direct(reference_srt_file, hypothesis_srt_file):
    if reference_srt_file is None:
        return "Veuillez uploader le SRT de référence.", ""
    if hypothesis_srt_file is None:
        return "Veuillez uploader le SRT généré.", ""

    try:
        with open(reference_srt_file.name, "r", encoding="utf-8") as f:
            reference_srt = f.read()
        with open(hypothesis_srt_file.name, "r", encoding="utf-8") as f:
            hypothesis_srt = f.read()
    except Exception as e:
        return f"Erreur lecture fichiers : {e}", ""

    try:
        report = compare_srt(reference_srt, hypothesis_srt)
    except Exception as e:
        return f"Erreur analyse : {e}", ""

    return format_metrics("Métriques", report.wer_result, hypothesis_label="Généré"), report.diff_html


def analyze_gladia(media_files, ref_srt_files, vocab_file=None, denoise_profile="Aucun"):

    logs = []

    def log(msg):
        logs.append(msg)
        print(msg)
        return "\n".join(logs)

    ui = {
        "gladia_metrics": "", "gladia_cv_metrics": "", "llm_metrics": "",
        "diff_gladia": "", "diff_gladia_cv": "", "diff_llm": "", "diff_llm_vs_gladia": "",
        "dl_gladia": None, "dl_gladia_cv": None, "dl_llm": None,
        "dl_sub": None, "dl_ins": None, "dl_del": None,
    }

    def emit(msg):
        return (
            log(msg),
            ui["gladia_metrics"], ui["gladia_cv_metrics"], ui["llm_metrics"],
            ui["diff_gladia"], ui["diff_gladia_cv"], ui["diff_llm"], ui["diff_llm_vs_gladia"],
            ui["dl_gladia"], ui["dl_gladia_cv"], ui["dl_llm"],
            ui["dl_sub"], ui["dl_ins"], ui["dl_del"],
        )

    base_vocabulary, vocab_status = load_vocabulary(vocab_file)

    errors = config.validate()
    if errors:
        yield emit("\n".join(errors))
        return

    if not media_files:
        yield emit("⚠️ Veuillez uploader au moins un fichier audio/vidéo.")
        return
    if not ref_srt_files:
        yield emit("⚠️ Veuillez uploader les SRT de référence.")
        return

    yield emit(f"🔍 Vérification des fichiers... | {vocab_status}")

    pairs, unmatched = match_files_by_name(media_files, ref_srt_files)
    if unmatched:
        yield emit(f"⚠️ Aucun SRT trouvé pour : {', '.join(unmatched)}")
        return
    if len(pairs) != len(ref_srt_files):
        yield emit("⚠️ Le nombre de fichiers audio et de SRT ne correspond pas.")
        return

    ref_srt_contents = []
    for _, ref_srt_file in pairs:
        with open(ref_srt_file.name, "r", encoding="utf-8") as f:
            ref_srt_contents.append(f.read())

    yield emit("📚 Extraction du vocabulaire depuis les SRT de référence...")
    extracted = extract_vocabulary_from_srts(ref_srt_contents, min_freq=2)
    if base_vocabulary:
        custom_vocabulary, added = merge_vocabularies(base_vocabulary, extracted)
        yield emit(f"✅ Vocabulaire : {len(base_vocabulary)} termes de base + {added} extraits des SRT = {len(custom_vocabulary)} total")
    else:
        custom_vocabulary = extracted
        yield emit(f"✅ Vocabulaire extrait des SRT : {len(custom_vocabulary)} termes")

    all_gladia_srt, all_gladia_cv_srt, all_ref_srt, labels = [], [], [], []

    try:
        client = GladiaClient()
        for i, (media, ref_srt_file) in enumerate(pairs):
            name = os.path.basename(media.name)
            labels.append(os.path.splitext(name)[0])

            denoise = None if denoise_profile == "Aucun" else denoise_profile

            yield emit(f"🎙️ [{i+1}/{len(pairs)}] Transcription Gladia en cours : {name}...")
            gladia_srt = client.transcribe(media.name, denoise=denoise)
            all_gladia_srt.append(gladia_srt)
            # yield emit(f"🎙️ Transcription Gladia CV + vocabulaire ({i+1}/{len(pairs)}) : {name}...")
            # all_gladia_cv_srt.append(client.transcribe(media.name, custom_vocabulary, denoise=denoise))
            all_gladia_cv_srt.append(gladia_srt)
            all_ref_srt.append(ref_srt_contents[i])

            ref_text = srt_file_to_plain_text(ref_srt_contents[i])
            wer_g = compute_wer(ref_text, srt_file_to_plain_text(gladia_srt))
            partial_report = compare_four_way(all_ref_srt, all_gladia_srt, all_gladia_cv_srt, all_gladia_srt, labels)
            ui["gladia_metrics"] = format_metrics(f"Métriques Gladia vs Référence ({i+1}/{len(pairs)} fichiers)", partial_report.wer_gladia, hypothesis_label="Gladia")
            ui["dl_gladia"] = srt_to_tempfile("\n\n".join(all_gladia_srt), "_gladia.srt")
            yield emit(f"✅ [{i+1}/{len(pairs)}] Gladia terminé : {name} | WER={wer_g.wer:.1%} ({wer_g.ref_word_count} mots ref, {wer_g.substitutions} sub, {wer_g.insertions} ins, {wer_g.deletions} del)")
    except Exception as e:
        yield emit(f"❌ Erreur Gladia : {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    try:
        llm = get_llm_client()
    except Exception as e:
        yield emit(f"❌ Erreur LLM init : {type(e).__name__}: {e}")
        return

    all_llm_srt = []
    usage = None

    try:
        for i, (gladia_s, gladia_cv_s, ref_s) in enumerate(zip(all_gladia_srt, all_gladia_cv_srt, all_ref_srt)):
            ref_text = srt_file_to_plain_text(ref_s)
            wer_g = compute_wer(ref_text, srt_file_to_plain_text(gladia_s)).wer
            wer_cv = compute_wer(ref_text, srt_file_to_plain_text(gladia_cv_s)).wer
            best_srt = gladia_s if wer_g <= wer_cv else gladia_cv_s
            best_label = "Gladia" if wer_g <= wer_cv else "Gladia CV"

            rag_passages = rag.find_similar(srt_file_to_plain_text(best_srt), n=6)
            yield emit(f"🤖 [{i+1}/{len(pairs)}] Correction LLM depuis {best_label} (WER={min(wer_g, wer_cv):.1%}) | RAG : {len(rag_passages)} passages récupérés...")

            corrected, usage = correct_srt(best_srt, llm, glossary=glossary or None)
            all_llm_srt.append(corrected)

            ref_text = srt_file_to_plain_text(ref_s)
            wer_llm = compute_wer(ref_text, srt_file_to_plain_text(corrected))
            partial_report = compare_four_way(all_ref_srt[:i+1], all_gladia_srt[:i+1], all_gladia_cv_srt[:i+1], all_llm_srt, labels[:i+1])
            ui["llm_metrics"] = format_metrics(f"Métriques LLM vs Référence ({i+1}/{len(pairs)} fichiers)", partial_report.wer_llm, hypothesis_label="LLM")
            ui["dl_llm"] = srt_to_tempfile("\n\n".join(all_llm_srt), "_llm.srt")
            yield emit(f"✅ [{i+1}/{len(pairs)}] LLM terminé | WER={wer_llm.wer:.1%} ({wer_llm.substitutions} sub, {wer_llm.insertions} ins, {wer_llm.deletions} del) | {usage}")
    except Exception as e:
        yield emit(f"❌ Erreur LLM : {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    yield emit("📊 Calcul des métriques finales et génération des diffs...")

    try:
        report = compare_four_way(all_ref_srt, all_gladia_srt, all_gladia_cv_srt, all_llm_srt, labels)
    except Exception as e:
        yield emit(f"❌ Erreur analyse : {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    ui["gladia_metrics"] = format_metrics("Métriques Gladia vs Référence", report.wer_gladia, hypothesis_label="Gladia")
    ui["gladia_cv_metrics"] = format_metrics("Métriques Gladia CV vs Référence", report.wer_gladia_cv, hypothesis_label="Gladia CV")
    ui["llm_metrics"] = format_metrics("Métriques LLM vs Référence", report.wer_llm, hypothesis_label="LLM")
    ui["diff_gladia"] = report.diff_gladia_html
    ui["diff_gladia_cv"] = report.diff_gladia_cv_html
    ui["diff_llm"] = report.diff_llm_html
    ui["diff_llm_vs_gladia"] = report.diff_llm_vs_gladia_html
    ui["dl_gladia"] = srt_to_tempfile("\n\n".join(all_gladia_srt), "_gladia.srt")
    ui["dl_gladia_cv"] = srt_to_tempfile("\n\n".join(all_gladia_cv_srt), "_gladia_cv.srt")
    ui["dl_llm"] = srt_to_tempfile("\n\n".join(all_llm_srt), "_llm.srt")
    ui["dl_sub"] = substitutions_to_tempfile(report.wer_gladia.substitution_examples)
    ui["dl_ins"] = list_to_tempfile(report.wer_gladia.insertion_examples, "_insertions.txt")
    ui["dl_del"] = list_to_tempfile(report.wer_gladia.deletion_examples, "_suppressions.txt")

    yield emit(f"✅ Terminé ! Gladia WER={report.wer_gladia.wer:.1%} | LLM WER={report.wer_llm.wer:.1%} | {usage}")


def add_media(new_files, existing):
    existing = existing or []
    if new_files:
        existing_names = {os.path.basename(f.name) for f in existing}
        new_unique = [f for f in new_files if os.path.basename(f.name) not in existing_names]
        existing = existing + new_unique
    names = "\n".join(f"- {os.path.basename(f.name)}" for f in existing)
    return None, existing, names


def add_srt(new_files, existing):
    existing = existing or []
    if new_files:
        existing_names = {os.path.basename(f.name) for f in existing}
        new_unique = [f for f in new_files if os.path.basename(f.name) not in existing_names]
        existing = existing + new_unique
    names = "\n".join(f"- {os.path.basename(f.name)}" for f in existing)
    return None, existing, names


def reset_uploads():
    return None, None, [], [], "", ""


_, default_vocab_status = load_vocabulary()

with gr.Blocks(title="Transcription Analyzer") as demo:
    gr.Markdown("# Transcription Analyzer")

    with gr.Tabs():

        with gr.Tab("Comparaison directe"):
            gr.Markdown("Comparez deux fichiers SRT sans passer par Gladia.")
            with gr.Row():
                direct_ref = gr.File(label="SRT référence (corrigé)", file_types=[".srt"])
                direct_hyp = gr.File(label="SRT généré (prédiction)", file_types=[".srt"])
            with gr.Row():
                direct_btn = gr.Button("Analyser", variant="primary")
                gr.ClearButton(components=[direct_ref, direct_hyp], value="🗑️ Réinitialiser")
            with gr.Tabs():
                with gr.Tab("Métriques"):
                    direct_metrics = gr.Markdown()
                with gr.Tab("Diff texte"):
                    direct_diff = gr.HTML()

        with gr.Tab("Transcription Gladia"):
            gr.Markdown("Transcrit via Gladia, corrige via LLM, compare les trois.")
            media_state = gr.State([])
            srt_state = gr.State([])
            with gr.Row():
                media_input = gr.File(
                    label="🎵 Glisser-déposer les fichiers audio / vidéo",
                    file_count="multiple",
                )
                srt_input = gr.File(
                    label="📄 Glisser-déposer les SRT référence",
                    file_types=[".srt"],
                    file_count="multiple",
                )
            with gr.Row():
                media_names = gr.Textbox(label="Fichiers audio ajoutés", interactive=False, lines=3)
                srt_names = gr.Textbox(label="SRT ajoutés", interactive=False, lines=3)
            with gr.Row():
                vocab_input = gr.File(
                    label="📚 Vocabulaire personnalisé (optionnel — remplace vocabulary.json)",
                    file_types=[".json"],
                )
                vocab_status = gr.Textbox(
                    label="Statut vocabulaire",
                    value=default_vocab_status,
                    interactive=False,
                    lines=1,
                )
            with gr.Row():
                denoise_input = gr.Dropdown(
                    label="🎛️ Réduction de bruit",
                    choices=["Aucun", "Auto", "léger", "modéré", "fort", "extrême"],
                    value="Aucun",
                    info="Auto = détecte le niveau de bruit | léger = voix intérieure | modéré = légère brise | fort = vent soutenu | extrême = vent fort",
                )
            with gr.Row():
                gladia_btn = gr.Button("Transcrire & Analyser", variant="primary")
                reset_btn = gr.Button("🗑️ Réinitialiser", variant="secondary")
            status_box = gr.Textbox(label="Statut", interactive=False, lines=12, max_lines=50)
            with gr.Tabs():
                with gr.Tab("Métriques Gladia"):
                    gladia_metrics = gr.Markdown()
                with gr.Tab("Métriques Gladia CV"):
                    gladia_cv_metrics = gr.Markdown()
                with gr.Tab("Métriques LLM"):
                    llm_metrics = gr.Markdown()
                with gr.Tab("Diff Gladia vs Référence"):
                    diff_gladia = gr.HTML()
                with gr.Tab("Diff Gladia CV vs Référence"):
                    diff_gladia_cv = gr.HTML()
                with gr.Tab("Diff LLM vs Référence"):
                    diff_llm = gr.HTML()
                with gr.Tab("Diff LLM vs Gladia"):
                    diff_llmVSgladia = gr.HTML()
            with gr.Row():
                download_gladia = gr.File(label="SRT Gladia")
                download_gladia_cv = gr.File(label="SRT Gladia CV")
                download_llm = gr.File(label="SRT LLM")
            with gr.Row():
                download_substitutions = gr.File(label="Substitutions (.csv)")
                download_insertions = gr.File(label="Insertions (.txt)")
                download_deletions = gr.File(label="Suppressions (.txt)")

    direct_btn.click(
        fn=analyze_direct,
        inputs=[direct_ref, direct_hyp],
        outputs=[direct_metrics, direct_diff],
    )

    media_input.change(
        fn=add_media,
        inputs=[media_input, media_state],
        outputs=[media_input, media_state, media_names],
    )

    srt_input.change(
        fn=add_srt,
        inputs=[srt_input, srt_state],
        outputs=[srt_input, srt_state, srt_names],
    )

    vocab_input.change(
        fn=refresh_vocab_status,
        inputs=[vocab_input],
        outputs=[vocab_status],
    )

    reset_btn.click(
        fn=reset_uploads,
        outputs=[media_input, srt_input, media_state, srt_state, media_names, srt_names],
    )

    gladia_btn.click(
        fn=analyze_gladia,
        inputs=[media_state, srt_state, vocab_input, denoise_input],
        outputs=[
            status_box,
            gladia_metrics,
            gladia_cv_metrics,
            llm_metrics,
            diff_gladia,
            diff_gladia_cv,
            diff_llm,
            diff_llmVSgladia,
            download_gladia,
            download_gladia_cv,
            download_llm,
            download_substitutions,
            download_insertions,
            download_deletions,
        ],
    )


if __name__ == "__main__":
    demo.launch()