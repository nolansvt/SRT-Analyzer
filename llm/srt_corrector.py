from llm.base import LLMClient, TokenUsage

def _build_prompt(srt_chunk: str, glossary_block: str) -> str:
    return f"""Tu es un outil de correction pour transcription automatique française.

            TON JOB : corriger le texte transcrit sur deux niveaux.

            1. ERREURS PHONÉTIQUES STT — quand le système a entendu un mot et en a écrit un autre qui sonne pareil ou proche :
            - "ossière" → "aussière" (même son, mauvaise graphie)
            - "matage" → "mâtage" (accent manquant sur terme technique)
            - "gération" → "giration" (déformation phonétique)
            - "j18" → "G18" (lettre phonétiquement proche)
            - "road shield" → "rothschild" (nom propre déformé)
            - "plume" → "clean" (mot anglais mal entendu)

            2. ERREURS LINGUISTIQUES — corriger normalement :
            - Fautes d'orthographe ("aprè" → "après", "cela" → "ça" si contexte oral)
            - Accords grammaticaux ("les pièce" → "les pièces")
            - Conjugaisons incorrectes
            - Accents manquants ("a" → "à" quand c'est une préposition)

            CE QUE TU NE CORRIGES PAS :
            - Style oral et registre familier ("t'as", "c'est pas", "on va y aller") — c'est voulu
            - Ponctuation manquante
            - Reformulation de phrases maladroites mais compréhensibles

            {glossary_block}RÈGLES ABSOLUES :
            - Ne touche pas aux timestamps ni aux numéros de blocs SRT
            - Retourne UNIQUEMENT le SRT corrigé, sans explication ni markdown
            - En cas de doute sur une correction, abstiens-toi
            - Chaque segment est INDÉPENDANT. Ne complète JAMAIS une phrase tronquée avec du texte d'autres segments. Corrige uniquement les mots déjà présents dans le segment, n'en ajoute aucun nouveau.

            SRT à corriger :
            {srt_chunk}"""


def _split_srt(srt_content: str, batch_size: int = 50) -> list[str]:
    blocks = [b for b in srt_content.strip().split('\n\n') if b.strip()]
    batches = []
    for i in range(0, len(blocks), batch_size):
        batches.append('\n\n'.join(blocks[i:i + batch_size]))
    return batches


def correct_srt(srt_content: str, llm_client: LLMClient, reference_srt: str | None = None, glossary: list[str] | None = None) -> tuple[str, TokenUsage]:
    print("Préparation du prompt pour la correction SRT...")

    glossary_block = ""
    if glossary:
        glossary_block = (
            "Ces termes nautiques existent et peuvent avoir été mal transcrits phonétiquement. "
            "Si tu identifies un mot proche de l'un d'eux, corrige-le :\n"
            + ", ".join(glossary) + "\n\n"
        )

    if reference_srt:
        print("Utilisation de la SRT de référence pour guider la correction...")
        prompt = f"""Tu es un expert en post-traitement de transcription audio française.

        Tu reçois deux fichiers SRT :
        - TRANSCRIPTION : générée automatiquement par un STT
        - RÉFÉRENCE : corrigée par un humain

        Ta tâche : comparer la TRANSCRIPTION et la RÉFÉRENCE segment par segment. Pour chaque différence, pose-toi cette unique question :

        "Est-ce la même information écrite différemment, ou des mots différents ?"

        SI C'EST LA MÊME INFORMATION ÉCRITE DIFFÉREMMENT → corrige dans la TRANSCRIPTION en adoptant le format de la RÉFÉRENCE.
        SI CE SONT DES MOTS DIFFÉRENTS → garde le texte de la TRANSCRIPTION tel quel.

        RÈGLES ABSOLUES :
        - Tu pars toujours de la TRANSCRIPTION
        - Tu ne changes jamais les timestamps ni la structure SRT
        - Retourne UNIQUEMENT le SRT corrigé, sans explication ni balise markdown

        TRANSCRIPTION :
        {srt_content}

        RÉFÉRENCE :
        {reference_srt}"""

        corrected, usage = llm_client.generate_with_usage(prompt)
        return corrected, usage

    batches = _split_srt(srt_content)
    print(f"SRT découpé en {len(batches)} lots")

    corrected_parts = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    for i, batch in enumerate(batches):
        print(f"Correction lot {i + 1}/{len(batches)}...")
        prompt = _build_prompt(batch, glossary_block)
        corrected, usage = llm_client.generate_with_usage(prompt)
        corrected_parts.append(corrected.strip())
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        total_cost += usage.estimated_cost_usd

    combined = '\n\n'.join(corrected_parts)
    combined_usage = TokenUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        estimated_cost_usd=total_cost,
    )

    return combined, combined_usage