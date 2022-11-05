import warnings

from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# English to French
HELSINKI_ENGLISH_TO_FRENCH = 'Helsinki-NLP/opus-mt-en-fr'
MODEL_EN_FR = AutoModelForSeq2SeqLM.from_pretrained(HELSINKI_ENGLISH_TO_FRENCH)
TOKENIZER_EN_FR = AutoTokenizer.from_pretrained(HELSINKI_ENGLISH_TO_FRENCH)
TRANSLATION_EN_FR = pipeline("translation_en_to_fr", model=MODEL_EN_FR, tokenizer=TOKENIZER_EN_FR)

# French to German
HELSINKI_FRENCH_TO_GERMAN = 'Helsinki-NLP/opus-mt-fr-de'
MODEL_FR_DE = AutoModelForSeq2SeqLM.from_pretrained(HELSINKI_FRENCH_TO_GERMAN)
TOKENIZER_FR_DE = AutoTokenizer.from_pretrained(HELSINKI_FRENCH_TO_GERMAN)
TRANSLATION_FR_DE = pipeline("translation_fr_to_de", model=MODEL_FR_DE, tokenizer=TOKENIZER_FR_DE)

# German to English
HELSINKI_GERMAN_TO_ENGLISH = 'Helsinki-NLP/opus-mt-de-en'
MODEL_DE_EN = AutoModelForSeq2SeqLM.from_pretrained(HELSINKI_GERMAN_TO_ENGLISH)
TOKENIZER_DE_EN = AutoTokenizer.from_pretrained(HELSINKI_GERMAN_TO_ENGLISH)
TRANSLATION_DE_EN = pipeline("translation_de_to_en", model=MODEL_DE_EN, tokenizer=TOKENIZER_DE_EN)

# Translation Pipelines Dictionary
PRE_TRAINED_MULTI_LINGUAL_TRANSLATION_PIPELINE = {
    'en-fr': TRANSLATION_EN_FR,
    'fr-de': TRANSLATION_FR_DE,
    'de-en': TRANSLATION_DE_EN
}


def _get_translation(sentence, model_pipeline) -> str:
    """
    Converts given sentence from one language to another using pre-trained Helsinki Models
    :param sentence: string in FROM language of model pipeline
    :param model_pipeline: Pre-Trained language model pipeline
    :return: string in TO language of model pipeline
    """
    if not sentence:
        return None
    warnings.filterwarnings("ignore")
    translated_text_result = model_pipeline(sentence, max_length=512)
    if translated_text_result:
        return translated_text_result[0]['translation_text']

    return None


def back_translation(sentence) -> str:
    """
    Uses Back-Translation from English - French - German - English
    :param sentence: string in english
    :return: string in english or None
    """
    if type(sentence) != str:  # Only string objects allowed
        return None
    translated_sentence = sentence
    for key, a_pipeline in PRE_TRAINED_MULTI_LINGUAL_TRANSLATION_PIPELINE.items():
        translated_sentence = _get_translation(translated_sentence, a_pipeline)

    return translated_sentence