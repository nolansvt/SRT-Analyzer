import gradio as gr
from config import config
from transcription.gladia_client import GladiaClient
from analysis.comparator import compare_srt
from analysis.wer import WERResult
from llm.factory import get_llm_client


def build_llm_prompt(wer_result: WERResult) -> str:
    sub_examples = "\n".join(
        f'  - "{r}" → "{h}"' for r, h in wer_result.substitution_examples[:5]
    )
    ins_examples = "\n".join(f'  - "{w}"' for w in wer_result.insertion_examples[:5])
    del_examples = "\n".join(f'  - "{w}"' for w in wer_result.deletion_examples[:5])

    return f"""Tu es un expert en qualité de transcription audio.

Métriques de la transcription automatique comparée à la référence corrigée :

- WER : {wer_result.wer:.1%}
- Substitutions : {wer_result.substitutions}
- Insertions : {wer_result.insertions}
- Suppressions : {wer_result.deletions}
- Mots référence : {wer_result.ref_word_count}
- Mots prédits : {wer_result.hyp_word_count}

Exemples de substitutions (référence -> prédiction) :
{sub_examples or "  Aucune"}

Exemples d'insertions :
{ins_examples or "  Aucune"}

Exemples de suppressions :
{del_examples or "  Aucune"}

Fournis une analyse structurée en 3 parties :
1. Résumé de la qualité globale (2-3 phrases)
2. Types d'erreurs les plus fréquentes
3. Suggestions concrètes d'amélioration
"""


def format_metrics(wer_result: WERResult) -> str:
    lines = [
        "## Métriques\n",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| WER (Word Error Rate) | **{wer_result.wer:.1%}** |",
        f"| MER (Match Error Rate) | {wer_result.mer:.1%} |",
        f"| WIL (Word Information Lost) | {wer_result.wil:.1%} |",
        f"| Mots référence (nombre de mots dans le texte de référence) | {wer_result.ref_word_count} |",
        f"| Mots prédits (nombre de mots dans la prédiction) | {wer_result.hyp_word_count} |",
        "\n## Erreurs\n",
        "| Type | Nombre |",
        "|---|---|",
        f"| Substitutions (mots remplacés) | {wer_result.substitutions} |",
        f"| Insertions (mots ajoutés) | {wer_result.insertions} |",
        f"| Suppressions (mots supprimés) | {wer_result.deletions} |",
        ]
    if wer_result.substitution_examples:
        lines.append("\n### Exemples de substitutions")
        for r, h in wer_result.substitution_examples:
            lines.append(f'- **"{r}"** → "{h}"')
    if wer_result.insertion_examples:
        lines.append("\n### Exemples d'insertions")
        for w in wer_result.insertion_examples:
            lines.append(f'- "{w}"')
    if wer_result.deletion_examples:
        lines.append("\n### Exemples de suppressions")
        for w in wer_result.deletion_examples:
            lines.append(f'- "{w}"')
    return "\n".join(lines)


def run_analysis(reference_srt: str, hypothesis_srt: str):
    try:
        report = compare_srt(reference_srt, hypothesis_srt)
    except Exception as e:
        return f"Erreur analyse : {e}", "", ""

    metrics_md = format_metrics(report.wer_result)

    try:
        llm = get_llm_client()
        llm_analysis = llm.generate(build_llm_prompt(report.wer_result))
    except Exception as e:
        llm_analysis = f"Analyse LLM indisponible : {e}"

    return metrics_md, llm_analysis, report.diff_html


def analyze_gladia(media_file, reference_srt_file):
    errors = config.validate()
    if errors:
        return "\n".join(errors), "", ""
    if media_file is None:
        return "Veuillez uploader un fichier audio/vidéo.", "", ""
    if reference_srt_file is None:
        return "Veuillez uploader un fichier SRT de référence.", "", ""

    try:
        with open(reference_srt_file.name, "r", encoding="utf-8") as f:
            reference_srt = f.read()
    except Exception as e:
        return f"Erreur lecture SRT référence : {e}", "", ""

    try:
        client = GladiaClient()
        generated_srt = client.transcribe(media_file.name)
    except Exception as e:
        return f"Erreur Gladia : {e}", "", ""

    return run_analysis(reference_srt, generated_srt)


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

    return run_analysis(reference_srt, hypothesis_srt)


def results_block():
    with gr.Tabs():
        with gr.Tab("Métriques"):
            metrics = gr.Markdown()
        with gr.Tab("Analyse LLM"):
            llm = gr.Markdown()
        with gr.Tab("Diff texte"):
            diff = gr.HTML()
    return metrics, llm, diff


with gr.Blocks(title="Transcription Analyzer") as demo:
    gr.Markdown("# Transcription Analyzer")

    with gr.Tabs():
        with gr.Tab("Comparaison directe"):
            gr.Markdown("Comparez deux fichiers SRT sans passer par Gladia.")
            with gr.Row():
                direct_ref = gr.File(label="SRT référence (corrigé)", file_types=[".srt"])
                direct_hyp = gr.File(label="SRT généré (prédiction)", file_types=[".srt"])
            direct_btn = gr.Button("Analyser", variant="primary")
            direct_metrics, direct_llm, direct_diff = results_block()

        with gr.Tab("Transcription Gladia"):
            gr.Markdown("Transcrit un fichier audio/vidéo via Gladia puis compare au SRT de référence.")
            with gr.Row():
                media_input = gr.File(label="Fichier audio / vidéo", file_types=["audio", "video"])
                srt_input = gr.File(label="SRT référence (.srt)", file_types=[".srt"])
            gladia_btn = gr.Button("Transcrire & Analyser", variant="primary")
            gladia_metrics, gladia_llm, gladia_diff = results_block()

    direct_btn.click(
        fn=analyze_direct,
        inputs=[direct_ref, direct_hyp],
        outputs=[direct_metrics, direct_llm, direct_diff],
    )

    gladia_btn.click(
        fn=analyze_gladia,
        inputs=[media_input, srt_input],
        outputs=[gladia_metrics, gladia_llm, gladia_diff],
    )


if __name__ == "__main__":
    demo.launch()