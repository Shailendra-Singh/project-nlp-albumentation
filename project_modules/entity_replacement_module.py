import re
import gensim.downloader as api
import numpy as np
import spacy
import stanza
import tensorflow_hub as tf_hub
from nltk.corpus import wordnet

WORDNET = wordnet
PATTERN_STR = r'[^project_modules-zA-Z\s]+'  # ^ char negates the pattern that follows it.
CLEAN_TEXT_PATTERN = re.compile(PATTERN_STR)
SP = spacy.load("en_core_web_sm")
STANZA_CONSTITUENCY_PIPELINE = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
USE_MODEL_URL = "/home/shailendrasingh/universal-sentence-encoder/"

# Try to use locally cached USE model, else load it from source URL.
try:
    USE_MODEL = tf_hub.load(USE_MODEL_URL)
except RuntimeError("'Universal Sentence Encoder' module not found."):
    USE_MODEL = tf_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

PRE_TRAINED_WORD2VEC = api.load('glove-wiki-gigaword-300')


def _get_semantic_similarity(word1, word2) -> float:
    """
    Calculates semantic similarity between two given words.
    semantic similarity = 1 - argcos(cosine similiraty) / π

    Above technique is implemented as described in: paper -	arXiv:1909.00088

    :param word1: string
    :param word2: string
    :return: float
    """
    word1_embedding = USE_MODEL([word1])
    word2_embedding = USE_MODEL([word2])
    cos_sims = np.inner(word1_embedding, word2_embedding)
    arc_cos = np.arccos(cos_sims)
    semantic_similarity = 1 - (arc_cos / np.pi)
    return semantic_similarity[0][0]


def get_most_semantic_similar_word(potential_targets, replacement_entity) -> str:
    """
    Returns the most semantic similar word based on semantic similarity score.
    :param potential_targets: list of potential targets for Replacement Entity
    :param replacement_entity: Replacement Entity string
    :return: text
    """
    target_word, score = None, -1
    for target in potential_targets:
        semantic_score = _get_semantic_similarity(target, replacement_entity)
        if semantic_score > score:
            score = semantic_score
            target_word = target

    return target_word


def get_constituency_dictionary(text, use_upos=False) -> dict:
    """
    Returns parts of speech constituents' dictionary for project_modules given text.

    Kristina Toutanova, Dan Klein, Christopher D. Man-
    ning, and Yoram Singer. 2003. Feature-rich part-of-
    speech tagging with a cyclic dependency network.
    In Proceedings of the 2003 Conference of the North
    American Chapter of the Association for Computa-
    tional Linguistics on Human Language Technology -
    Volume 1, NAACL ’03, page 173–180, USA. Asso-
    ciation for Computational Linguistics.

    (https://dl.acm.org/doi/10.3115/1073445.1073478)

    :param text: string
    :return: dict
    """
    transformed_text = STANZA_CONSTITUENCY_PIPELINE(text)
    pos_dict = {}
    for sent in transformed_text.sentences:
        for word in sent.words:
            pos = word.xpos
            if use_upos:
                pos = word.upos
            xpos_list = pos_dict.get(pos, [])
            xpos_list.append((word.text))
            pos_dict[pos] = xpos_list

    # Explicitly  add NN key to avoid key not found error.
    if not use_upos and "NN" not in pos_dict.keys():
        pos_dict["NN"] = [None]
    return pos_dict


def get_noun_phrases(text) -> str:
    """
    Used spacy to extract noun-phrases.
    :param text: string
    :return: string
    """
    corpus = SP(text)
    # Extract noun-phrases
    noun_phrases = [w.text for w in corpus.noun_chunks if len(w.text) > 2]
    return noun_phrases


def generate_replacement_noun_phrase(target_noun_phrase, replacement_entity) -> str:
    """
    Replaces the target entity in source noun phrase.
    :param target_noun_phrase: source entity noun phrase
    :param replacement_entity: target entity noun
    :return:new noun phrase
    """
    word_to_replace = get_constituency_dictionary(target_noun_phrase)["NN"][0]  # first noun
    if not word_to_replace:
        word_to_replace = target_noun_phrase  # Replace the whole entity if no nouns were found

    return target_noun_phrase.replace(word_to_replace, replacement_entity)


def _get_random_items(np_array, num_items=1, with_replacement=False) -> list:
    """
    Selects random item from project_modules numpy array.
    :param np_array: source numpy array
    :param num_items: number of items to fetch
    :param with_replacement: replace selected samples
    :return: list
    """
    len_np_array = len(np_array)
    if num_items > len_np_array:
        num_items = len_np_array
    index = np.random.choice(np_array.shape[0], num_items, replace=with_replacement)
    return np_array[index].tolist()


def get_similar_random_entity(entity):
    """
    Generate similar entity from pre-trained model.
    :param entity: single string or iterable of string
    :return: string or iterable
    """
    if not isinstance(entity, str) and hasattr(entity, '__iter__'):
        all_items = []
        for item in entity:
            most_similar_entities = []
            try:
                for word, score in PRE_TRAINED_WORD2VEC.most_similar(item):
                    if len(word) > 3:
                        most_similar_entities.append(word)
            except:
                continue
            if most_similar_entities:
                most_similar_entities = np.array(most_similar_entities)
                all_items.append(_get_random_items(most_similar_entities)[0])
        return all_items
    else:
        most_similar_entities = []
        try:
            for word, score in PRE_TRAINED_WORD2VEC.most_similar(entity):
                if len(word) > 3:
                    most_similar_entities.append(word)
        except:
            pass
        if most_similar_entities:
            most_similar_entities = np.array(most_similar_entities)
            return _get_random_items(most_similar_entities)[0]
        else:
            return ''


def _get_onyms(word, find_synonym=True) -> set:
    """
    Returns synonyms or antonyms of given word.
    :param word: string
    :param find_synonym: True: returns synonym; False: returns antonym
    :return: set
    """
    onyma_set = set()
    for syn in WORDNET.synsets(word):
        for lemma in syn.lemmas():
            if find_synonym:
                onyma_set.add(lemma.name())
            else:
                if lemma.antonyms():
                    onyma_set.add(lemma.antonyms()[0].name())
    return onyma_set


def get_onyms_replacements(sentence, with_synonym=True) -> dict:
    """
    Gets dictionary of possible random replacements of either synonyms or antonyms
    for words that are Verbs, Adjectives, or Adverbs.
    :param sentence: string
    :param with_synonym: True: get synonym replacement; False: get antonym replacement
    :return: dict
    """
    sentence = sentence.lower()
    constituency_dictionary = get_constituency_dictionary(sentence, use_upos=True)
    onym_dictionary = dict()
    for k, v in constituency_dictionary.items():
        if k in ('VERB', 'ADJ', 'ADV'):
            for item in v:
                item_onyms = _get_onyms(item, find_synonym=with_synonym)
                if item_onyms:
                    potential_onym = _get_random_items(np.array(list(item_onyms)))
                    onym_dictionary[item] = potential_onym

    return onym_dictionary


def replace_onyms(sentence, onym_dictionary) -> str:
    """
    Replaces all occurrences of an onym keys with values.
    :param sentence: string
    :param onym_dictionary: antonym or synonym potential replacements.
    :return: string
    """
    new_sentence = sentence[:]
    for k, v in onym_dictionary.items():
        if k in sentence:
            if v:
                new_sentence = new_sentence.replace(k, v[0])

    return new_sentence
