import os
import tempfile
import gradio as gr
from config import config
from transcription.gladia_client import GladiaClient
from analysis.comparator import compare_srt, compare_three_way
from analysis.wer import WERResult
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
        lines.append(f"\n### Exemples de substitutions\n| Référence | {hypothesis_label} |\n|---|---|")
        for r, h in wer.substitution_examples:
            lines.append(f"| {r} | {h} |")
    if wer.insertion_examples:
        lines.append(f"\n### Mots ajoutés par {hypothesis_label} (insertions)")
        for w in wer.insertion_examples:
            lines.append(f"- {w}")
    if wer.deletion_examples:
        lines.append("\n### Mots manquants dans la transcription (suppressions)")
        for w in wer.deletion_examples:
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
        return (msg, "", "", "", "", None, None)

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

    all_gladia_srt, all_ref_srt = [], []

    try:
        client = GladiaClient()
        for i, (media, ref_srt_file) in enumerate(pairs):
            name = os.path.basename(media.name)
            yield s(f"🎙️ Transcription Gladia ({i+1}/{len(pairs)}) : {name}...")
            all_gladia_srt.append(client.transcribe(media.name))
            with open(ref_srt_file.name, "r", encoding="utf-8") as f:
                all_ref_srt.append(f.read())
    except Exception as e:
        yield s(f"❌ Erreur Gladia : {e}")
        return

    gladia_srt = "\n\n".join(all_gladia_srt)
    reference_srt = "\n\n".join(all_ref_srt)

    yield s("🤖 Correction LLM en cours...")

    try:
        llm = get_llm_client()
        llm_srt, usage = correct_srt(gladia_srt, llm)
    except Exception as e:
        yield s(f"❌ Erreur LLM : {e}")
        return

    yield s("📊 Analyse en cours...")

    try:
        report = compare_three_way(reference_srt, gladia_srt, llm_srt)
    except Exception as e:
        yield s(f"❌ Erreur analyse : {e}")
        return

    gladia_file = srt_to_tempfile(gladia_srt, "_gladia.srt")
    llm_file = srt_to_tempfile(llm_srt, "_llm_corrected.srt")

    yield (
        f"✅ Terminé ! | {usage}",
        format_metrics("Métriques Gladia vs Référence", report.wer_gladia, hypothesis_label="Gladia"),
        format_metrics("Métriques LLM vs Référence", report.wer_llm, hypothesis_label="LLM"),
        report.diff_gladia_html,
        report.diff_llm_html,
        gladia_file,
        llm_file,
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
                with gr.Tab("Métriques LLM"):
                    llm_metrics = gr.Markdown()
                with gr.Tab("Diff Gladia vs Référence"):
                    diff_gladia = gr.HTML()
                with gr.Tab("Diff LLM vs Référence"):
                    diff_llm = gr.HTML()
            with gr.Row():
                download_gladia = gr.File(label="Télécharger SRT Gladia")
                download_llm = gr.File(label="Télécharger SRT corrigé (LLM)")

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
        outputs=[status_box, gladia_metrics, llm_metrics, diff_gladia, diff_llm, download_gladia, download_llm],
    )


if __name__ == "__main__":
    demo.launch()