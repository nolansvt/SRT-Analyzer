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


def analyze_gladia(media_files, ref_srt_files):

    def s(msg):
        return (msg, "", "", "", "", "", "", None, None, None, None, None, None)

    custom_vocabulary = None
    if config.GLADIA_VOCABULARY_PATH and os.path.exists(config.GLADIA_VOCABULARY_PATH):
        with open(config.GLADIA_VOCABULARY_PATH, "r", encoding="utf-8") as f:
            custom_vocabulary = json.load(f)

    errors = config.validate()
    if errors:
        yield s("\n".join(errors))
        return

    if not media_files:
        yield s("⚠️ Veuillez uploader au moins un fichier audio/vidéo.")
        return
    if not ref_srt_files:
        yield s("⚠️ Veuillez uploader les SRT de référence.")
        return

    yield s("🔍 Vérification de la correspondance des fichiers...")

    pairs, unmatched = match_files_by_name(media_files, ref_srt_files)
    if unmatched:
        yield s(f"⚠️ Aucun SRT trouvé pour : {', '.join(unmatched)}")
        return
    if len(pairs) != len(ref_srt_files):
        yield s("⚠️ Le nombre de fichiers audio et de SRT ne correspond pas.")
        return

    all_gladia_srt, all_gladia_cv_srt, all_ref_srt, labels = [], [], [], []

    try:
        client = GladiaClient()
        for i, (media, ref_srt_file) in enumerate(pairs):
            name = os.path.basename(media.name)
            labels.append(os.path.splitext(name)[0])

            yield s(f"🎙️ Transcription Gladia ({i+1}/{len(pairs)}) : {name}...")
            all_gladia_srt.append(client.transcribe(media.name))

            if custom_vocabulary:
                yield s(f"🎙️ Transcription Gladia CV ({i+1}/{len(pairs)}) : {name}...")
                all_gladia_cv_srt.append(client.transcribe(media.name, custom_vocabulary))
            else:
                all_gladia_cv_srt.append(all_gladia_srt[-1])

            with open(ref_srt_file.name, "r", encoding="utf-8") as f:
                all_ref_srt.append(f.read())
    except Exception as e:
        yield s(f"❌ Erreur Gladia : {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    try:
        llm = get_llm_client()
    except Exception as e:
        yield s(f"❌ Erreur LLM init : {type(e).__name__}: {e}")
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

            yield s(f"🤖 Correction LLM ({i+1}/{len(pairs)}) depuis {best_label} (WER={min(wer_g, wer_cv):.1%})...")
            corrected, usage = correct_srt(best_srt, llm)
            all_llm_srt.append(corrected)
    except Exception as e:
        yield s(f"❌ Erreur LLM : {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    yield s("📊 Analyse en cours...")

    try:
        report = compare_four_way(all_ref_srt, all_gladia_srt, all_gladia_cv_srt, all_llm_srt, labels)
    except Exception as e:
        yield s(f"❌ Erreur analyse : {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    gladia_file = srt_to_tempfile("\n\n".join(all_gladia_srt), "_gladia.srt")
    gladia_cv_file = srt_to_tempfile("\n\n".join(all_gladia_cv_srt), "_gladia_cv.srt")
    llm_file = srt_to_tempfile("\n\n".join(all_llm_srt), "_llm.srt")
    sub_file = substitutions_to_tempfile(report.wer_gladia.substitution_examples)
    ins_file = list_to_tempfile(report.wer_gladia.insertion_examples, "_insertions.txt")
    del_file = list_to_tempfile(report.wer_gladia.deletion_examples, "_suppressions.txt")

    yield (
        f"✅ Terminé ! | {usage}",
        format_metrics("Métriques Gladia vs Référence", report.wer_gladia, hypothesis_label="Gladia"),
        format_metrics("Métriques Gladia CV vs Référence", report.wer_gladia_cv, hypothesis_label="Gladia CV"),
        format_metrics("Métriques LLM vs Référence", report.wer_llm, hypothesis_label="LLM"),
        report.diff_gladia_html,
        report.diff_gladia_cv_html,
        report.diff_llm_html,
        gladia_file,
        gladia_cv_file,
        llm_file,
        sub_file,
        ins_file,
        del_file,
    )


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
                gladia_btn = gr.Button("Transcrire & Analyser", variant="primary")
                reset_btn = gr.Button("🗑️ Réinitialiser", variant="secondary")
            status_box = gr.Textbox(label="Statut", interactive=False)
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

    reset_btn.click(
        fn=reset_uploads,
        outputs=[media_input, srt_input, media_state, srt_state, media_names, srt_names],
    )

    gladia_btn.click(
        fn=analyze_gladia,
        inputs=[media_state, srt_state],
        outputs=[
            status_box,
            gladia_metrics,
            gladia_cv_metrics,
            llm_metrics,
            diff_gladia,
            diff_gladia_cv,
            diff_llm,
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