import re
from analysis.srt_parser import srt_file_to_plain_text

FRENCH_COMMON = {
    # articles / déterminants
    "les", "des", "une", "cette", "cela", "celui", "celle", "ceux", "celles",
    "notre", "votre", "leurs", "quel", "quelle", "quels", "quelles",
    # pronoms
    "nous", "vous", "ils", "elles", "lui", "leur", "moi", "toi", "soi",
    "lequel", "laquelle", "dont", "quoi", "rien", "quelqu",
    # conjonctions / prépositions
    "mais", "donc", "comme", "parce", "puisque", "lorsque", "tandis",
    "selon", "malgré", "depuis", "avant", "après", "pendant", "entre",
    "vers", "sous", "sans", "lors", "chez", "quant", "aussi",
    # adverbes fréquents
    "très", "bien", "plus", "moins", "encore", "même", "aussi", "ainsi",
    "surtout", "plutôt", "environ", "seulement", "simplement", "exactement",
    "normalement", "directement", "rapidement", "facilement", "souvent",
    "vraiment", "maintenant", "toujours", "jamais", "déjà", "ailleurs",
    "autour", "dedans", "dehors", "dessus", "dessous", "devant", "derrière",
    "ensemble", "souvent", "parfois", "plusieurs", "beaucoup", "tellement",
    # verbes très courants
    "faire", "faut", "aller", "avoir", "être", "peut", "vais", "veux",
    "fait", "avait", "était", "sera", "serait", "aurait", "ferait",
    "dire", "voir", "savoir", "prendre", "venir", "mettre", "donner",
    "passer", "rester", "sembler", "permettre", "trouver", "arriver",
    "utiliser", "falloir", "devoir", "pouvoir", "vouloir", "pense",
    "sais", "vois", "vois", "aime", "aimer", "peux", "doit", "fais",
    "pense", "parle", "regarde", "passe", "prend", "reste", "tiens",
    "allons", "venez", "avons", "êtes", "mettez", "prenez",
    # noms trop génériques
    "chose", "choses", "moment", "endroit", "partie", "temps", "place",
    "façon", "point", "côté", "niveau", "type", "sorte", "genre",
    "monde", "pays", "personnes", "gens", "homme", "femme", "enfant",
    "hommes", "femmes", "enfants", "groupe", "équipe", "sport", "sports",
    "caméra", "photo", "vidéo", "image", "écran", "espace", "ligne",
    "minutes", "secondes", "heures", "jours", "semaines", "mois",
    "bateau", "bateaux", "voile", "voiles", "vent", "vents", "vague",
    "vitesse", "pression", "préparation", "physique", "énergie",
    "question", "réponse", "problème", "solution", "résultat",
    "première", "dernier", "dernier", "prochain", "suivant",
    # adjectifs courants
    "grand", "grande", "petit", "petite", "gros", "grosse", "long",
    "large", "fort", "vite", "loin", "près", "nouveau", "nouvelle",
    "vieux", "vieille", "jeune", "beau", "belle", "bon", "bonne",
    "mauvais", "mauvaise", "seul", "seule", "propre", "juste", "vrai",
    "faux", "libre", "plein", "vide", "ouvert", "fermé", "possible",
    "impossible", "important", "difficile", "facile", "nécessaire",
    "différent", "différente", "certain", "certaine", "super", "énorme",
    "incroyable", "formidable", "magnifique", "terrible", "parfait",
    # mots courants 4 lettres
    "avec", "pour", "dans", "sur", "par", "tout", "plus", "sont",
    "deux", "trois", "voilà", "voici", "parce", "moins", "quand",
    "alors", "dont", "donc", "mais", "leur", "vers", "sous", "sans",
    "très", "bien", "prêt", "même", "fois", "cent", "mois", "jour",
    "donc", "ouais", "okay", "voil", "cool", "rien", "trop", "faut",
    "autre", "après", "avant", "entre", "comme", "aussi", "quoi",
    "merci", "bonjour", "bonsoir", "allons", "allez", "voilà",
    "peu", "assez", "autant", "tant", "non", "oui", "peut",
    "chaque", "toute", "toutes", "tous", "plusieurs", "aucun", "aucune",
    "quelque", "quelques", "comment", "pourquoi",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{5,}", text)


def extract_vocabulary_from_srts(srt_contents: list[str], min_freq: int = 2) -> list[dict]:
    word_counts: dict[str, int] = {}

    for srt in srt_contents:
        text = srt_file_to_plain_text(srt)
        for word in _tokenize(text):
            normalized = word.lower()
            if normalized not in FRENCH_COMMON:
                word_counts[word] = word_counts.get(word, 0) + 1

    seen_lower: set[str] = set()
    entries = []

    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        if count < min_freq:
            continue
        lower = word.lower()
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        entries.append({"value": word, "intensity": 0.4})

    return entries


def merge_vocabularies(base: list[dict], extracted: list[dict]) -> list[dict]:
    existing_values = {e["value"].lower() for e in base}
    merged = list(base)
    added = 0
    for entry in extracted:
        if entry["value"].lower() not in existing_values:
            merged.append(entry)
            existing_values.add(entry["value"].lower())
            added += 1
    return merged, added
