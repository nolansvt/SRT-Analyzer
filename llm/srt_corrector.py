from llm.base import LLMClient, TokenUsage


def correct_srt(srt_content: str, llm_client: LLMClient, reference_srt: str | None = None, glossary: list[str] | None = None) -> tuple[str, TokenUsage]:
    print("Préparation du prompt pour la correction SRT...")

    glossary_block = ""
    if glossary:
        glossary_block = (
            "Ces termes nautiques existent et peuvent avoir été mal transcrits phonétiquement. "
            "Si tu identifies un mot proche de l'un d'eux, corrige-le :\n"
            + ", ".join(glossary) + "\n\n"
        )
    print(glossary_block)

    if reference_srt:
        print("reference")
        prompt = f"""Tu es un expert en post-traitement de transcription audio française.

        Tu reçois deux fichiers SRT :
        - TRANSCRIPTION : générée automatiquement par un STT
        - RÉFÉRENCE : corrigée par un humain

        Ta tâche : comparer la TRANSCRIPTION et la RÉFÉRENCE segment par segment. Pour chaque différence, pose-toi cette unique question :

        "Est-ce la même information écrite différemment, ou des mots différents ?"

        SI C'EST LA MÊME INFORMATION ÉCRITE DIFFÉREMMENT → corrige dans la TRANSCRIPTION en adoptant le format de la RÉFÉRENCE :
        - "4" vs "quatre" → même nombre, écrit différemment → corrige
        - "14h" vs "14 h" → même heure, espacement différent → corrige
        - "g18" vs "g 18" → même code, espacement différent → corrige
        - "nœuds" vs "noeuds" → même mot, typographie différente → corrige
        - "ça" vs "cela" → même mot, variante différente → corrige

        SI CE SONT DES MOTS DIFFÉRENTS → garde le texte de la TRANSCRIPTION tel quel :
        - "vous partez" vs "pouvez partir" → sens différent → garde "vous partez"
        - "il arrive" vs "elle repart" → phrases différentes → garde "il arrive"
        - "demain matin" vs "ce soir" → information différente → garde "demain matin"

        RÈGLES ABSOLUES :
        - Tu pars toujours de la TRANSCRIPTION
        - Tu ne changes jamais les timestamps ni la structure SRT
        - Retourne UNIQUEMENT le SRT corrigé, sans explication ni balise markdown

        TRANSCRIPTION :
        {srt_content}

        RÉFÉRENCE :
        {reference_srt}"""

    else:
        print("no reference")
        prompt = f"""Tu es un outil de correction phonétique pour transcription automatique française.

        TON SEUL JOB : identifier les mots qui ont été mal reconnus phonétiquement par le STT et les corriger.

        Une erreur STT phonétique c'est quand le système a entendu un mot et en a écrit un autre qui sonne pareil ou proche :
        - "ossière" → "aussière" (même son, mauvaise graphie)
        - "matage" → "mâtage" (accent manquant sur terme technique)
        - "gération" → "giration" (déformation phonétique)
        - "j18" → "G18" (lettre phonétiquement proche)
        - "road shield" → "rothschild" (nom propre déformé)
        - "plume" → "clean" (mot anglais mal entendu)

        CE QUE TU NE CORRIGES PAS :
        - Grammaire, conjugaison, accords
        - Ponctuation manquante
        - Style oral ("t'as", "c'est pas", "on va y aller")
        - Un mot compréhensible même s'il semble familier

        {glossary_block}RÈGLES ABSOLUES :
        - Ne touche pas aux timestamps ni aux numéros de blocs SRT
        - Retourne UNIQUEMENT le SRT corrigé, sans explication ni markdown
        - Si tu n'es pas sûr qu'un mot est une erreur STT, ne le touche pas

        SRT à corriger :
        {srt_content}"""

    return llm_client.generate_with_usage(prompt)