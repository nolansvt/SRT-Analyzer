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


def format_metrics(label: str, wer: WERResult) -> str:
    lines = [
        f"## {label}\n",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| WER | **{wer.wer:.1%}** |",
        f"| MER | {wer.mer:.1%} |",
        f"| WIL | {wer.wil:.1%} |",
        f"| Mots référence | {wer.ref_word_count} |",
        f"| Mots prédits | {wer.hyp_word_count} |",
        "\n| Type d'erreur | Nombre |",
        "|---|---|",
        f"| Substitutions | {wer.substitutions} |",
        f"| Insertions | {wer.insertions} |",
        f"| Suppressions | {wer.deletions} |",
    ]
    if wer.substitution_examples:
        lines.append("\n### Exemples de substitutions")
        for r, h in wer.substitution_examples:
            lines.append(f'- **"{r}"** → "{h}"')
    if wer.insertion_examples:
        lines.append("\n### Exemples d'insertions")
        for w in wer.insertion_examples:
            lines.append(f'- "{w}"')
    if wer.deletion_examples:
        lines.append("\n### Exemples de suppressions")
        for w in wer.deletion_examples:
            lines.append(f'- "{w}"')
    return "\n".join(lines)


def analyze_direct(reference_srt_file, hypothesis_srt_file):
    if reference_srt_file is None:
        return "Veuillez uploader le SRT de référence.", "", ""
    if hypothesis_srt_file is None:
        return "Veuillez uploader le SRT généré.", "", ""

    try:
        with open(reference_srt_file.name, "r", encoding="utf-8") as f:
            reference_srt = f.read()
        with open(hypothesis_srt_file.name, "r", encoding="utf-8") as f:
            hypothesis_srt = f.read()
    except Exception as e:
        return f"Erreur lecture fichiers : {e}", "", ""

    try:
        report = compare_srt(reference_srt, hypothesis_srt)
    except Exception as e:
        return f"Erreur analyse : {e}", "", ""

    return format_metrics("Métriques", report.wer_result), "", report.diff_html


def analyze_gladia(media_files, ref_srt_files):
    empty = ("", "", "", "", "", None, None)

    errors = config.validate()
    if errors:
        yield "\n".join(errors), *empty[1:]
        return

    if not media_files:
        yield "⚠️ Veuillez uploader au moins un fichier audio/vidéo.", *empty[1:]
        return
    if not ref_srt_files:
        yield "⚠️ Veuillez uploader les SRT de référence.", *empty[1:]
        return

    yield "🔍 Vérification de la correspondance des fichiers...", *empty[1:]

    pairs, unmatched = match_files_by_name(media_files, ref_srt_files)
    if unmatched:
        yield f"⚠️ Aucun SRT trouvé pour : {', '.join(unmatched)}", *empty[1:]
        return
    if len(pairs) != len(ref_srt_files):
        yield "⚠️ Le nombre de fichiers audio et de SRT ne correspond pas.", *empty[1:]
        return

    all_gladia_srt, all_ref_srt = [], []

    try:
        client = GladiaClient()
        for i, (media, ref_srt_file) in enumerate(pairs):
            name = os.path.basename(media.name)
            yield f"🎙️ Transcription Gladia ({i+1}/{len(pairs)}) : {name}...", *empty[1:]
            all_gladia_srt.append(client.transcribe(media.name))
            with open(ref_srt_file.name, "r", encoding="utf-8") as f:
                all_ref_srt.append(f.read())
    except Exception as e:
        yield f"❌ Erreur Gladia : {e}", *empty[1:]
        return

    gladia_srt = "\n\n".join(all_gladia_srt)
    reference_srt = "\n\n".join(all_ref_srt)

    yield "🤖 Correction LLM en cours...", *empty[1:]

    try:
        llm = get_llm_client()
        llm_srt = correct_srt(gladia_srt, llm)
    except Exception as e:
        yield f"❌ Erreur LLM : {e}", *empty[1:]
        return

    yield "📊 Analyse en cours...", *empty[1:]

    try:
        report = compare_three_way(reference_srt, gladia_srt, llm_srt)
    except Exception as e:
        yield f"❌ Erreur analyse : {e}", *empty[1:]
        return

    gladia_file = srt_to_tempfile(gladia_srt, "_gladia.srt")
    llm_file = srt_to_tempfile(llm_srt, "_llm_corrected.srt")

    metrics_gladia = format_metrics("Métriques Gladia vs Référence", report.wer_gladia)
    metrics_llm = format_metrics("Métriques LLM vs Référence", report.wer_llm)

    yield (
        "✅ Terminé !",
        metrics_gladia,
        metrics_llm,
        report.diff_gladia_html,
        report.diff_llm_html,
        gladia_file,
        llm_file,
    )


with gr.Blocks(title="Transcription Analyzer") as demo:
    gr.Markdown("# Transcription Analyzer")

    with gr.Tabs():

        with gr.Tab("Comparaison directe"):
            gr.Markdown("Comparez deux fichiers SRT sans passer par Gladia.")
            with gr.Row():
                direct_ref = gr.File(label="SRT référence (corrigé)", file_types=[".srt"])
                direct_hyp = gr.File(label="SRT généré (prédiction)", file_types=[".srt"])
            direct_btn = gr.Button("Analyser", variant="primary")
            with gr.Tabs():
                with gr.Tab("Métriques"):
                    direct_metrics = gr.Markdown()
                with gr.Tab("Diff texte"):
                    direct_diff = gr.HTML()

        with gr.Tab("Transcription Gladia"):
            gr.Markdown("Transcrit via Gladia, corrige via LLM, compare les trois.")
            with gr.Row():
                media_input = gr.File(
                    label="Fichiers audio / vidéo",
                    file_types=["audio", "video"],
                    file_count="multiple",
                )
                srt_input = gr.File(
                    label="SRT référence (même nom que le fichier audio)",
                    file_types=[".srt"],
                    file_count="multiple",
                )
            gladia_btn = gr.Button("Transcrire & Analyser", variant="primary")
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
        outputs=[direct_metrics, gr.Textbox(visible=False), direct_diff],
    )

    gladia_btn.click(
        fn=analyze_gladia,
        inputs=[media_input, srt_input],
        outputs=[status_box, gladia_metrics, llm_metrics, diff_gladia, diff_llm, download_gladia, download_llm],
    )


if __name__ == "__main__":
    demo.launch()