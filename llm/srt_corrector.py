from llm.base import LLMClient


def correct_srt(srt_content: str, llm_client: LLMClient) -> str:
    prompt = f"""Tu es un expert en transcription audio française.

Voici un fichier SRT généré automatiquement par un système de reconnaissance vocale.
Corrige uniquement les erreurs de transcription : mots mal reconnus, noms propres incorrects, erreurs grammaticales évidentes dues à la reconnaissance vocale.

RÈGLES STRICTES :
- Conserve exactement le même format SRT (numéros, timestamps, structure)
- Ne modifie PAS les timestamps
- Ne fusionne pas et ne découpes pas les segments
- Retourne UNIQUEMENT le contenu SRT corrigé, sans explication ni balise markdown

SRT à corriger :
{srt_content}"""

    return llm_client.generate(prompt)