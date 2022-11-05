import re

from project_modules.back_translation_module import back_translation
from project_modules.entity_replacement_module import get_noun_phrases, get_constituency_dictionary, \
    get_most_semantic_similar_word, generate_replacement_noun_phrase, CLEAN_TEXT_PATTERN, get_similar_random_entity, \
    get_onyms_replacements, replace_onyms


class CustomAugmentation:

    def __init__(self):
        self.original_sentence = ''

    def get_augmented_sentence(self, original_sentence, num_iter=2) -> str:
        """
        Generates an augmented text using Entity Replacement technique as described in

        Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant,
        Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. 2018. Universal
        sentence encoder for English. In Proceedings of the 2018 Conference on Empirical Methods in Nat-
        ural Language Processing: System Demonstrations, pages 169–174, Brussels, Belgium. Association for
        Computational Linguistics.

        (https://aclanthology.org/D18-2029/)

        :param original_sentence: string
        :param num_iter: number of iteration for replacement. Higher number will slow down the processing.
        -1 for max iteration
        :return: string
        """
        if not original_sentence:
            return RuntimeError("Invalid sentence string")

        if not self.original_sentence:
            self.original_sentence = original_sentence

        transformed_sentence = original_sentence.lower()
        transformed_sentence = re.sub(CLEAN_TEXT_PATTERN, '', transformed_sentence)

        noun_phrases = get_noun_phrases(transformed_sentence)
        if not noun_phrases:
            return 'NA'  # Nothing to augment

        potential_replacement_entities = get_constituency_dictionary(transformed_sentence)["NN"]
        if not potential_replacement_entities:
            return 'NA'  # No potential entity found

        if num_iter > 0:
            potential_replacement_entities = potential_replacement_entities[:num_iter]

        all_potential_replacement_entities = [get_similar_random_entity(entity)
                                              for entity in potential_replacement_entities if
                                              entity and len(entity) > 0]

        new_sentence = original_sentence
        for i, an_entity in enumerate(all_potential_replacement_entities):
            if i > num_iter:
                break
            try:
                noun_phrases = get_noun_phrases(new_sentence)
                semantic_noun_phrases = get_most_semantic_similar_word(noun_phrases, an_entity)
                replacement_noun_phrase = generate_replacement_noun_phrase(semantic_noun_phrases, an_entity)
                new_sentence = new_sentence.replace(semantic_noun_phrases, replacement_noun_phrase)
            except:
                return "NA"

        return new_sentence

    def augment_with_synonyms(self, sentence) -> str:
        """
        Replaces all Verbs, Adjectives, and Adverbs with their synonyms.

        This technique was described in:

        P. Liu, X. Wang, C. Xiang and W. Meng, "A Survey of Text Data Augmentation," 2020 International Conference on
        Computer Communication and Network Security (CCNS), 2020, pp. 191-195, doi: 10.1109/CCNS50731.2020.00049.

        :param sentence: str
        :return: str
        """
        if type(sentence) != str:
            return None
        onyms_dictionary = get_onyms_replacements(sentence, with_synonym=True)
        new_sentence = replace_onyms(sentence, onyms_dictionary)
        return new_sentence

    def augment_with_antonyms(self, sentence) -> str:
        """
        Replaces all Verbs, Adjectives, and Adverbs with their antonyms.

        This technique was described in:

        P. Liu, X. Wang, C. Xiang and W. Meng, "A Survey of Text Data Augmentation," 2020 International Conference on
        Computer Communication and Network Security (CCNS), 2020, pp. 191-195, doi: 10.1109/CCNS50731.2020.00049.

        :param sentence: str
        :return: str
        """
        if type(sentence) != str:
            return None
        onyms_dictionary = get_onyms_replacements(sentence, with_synonym=False)
        new_sentence = replace_onyms(sentence, onyms_dictionary)
        return new_sentence

    def augment_with_back_translation(self, sentence) -> str:
        """
        Uses Back - Translation technique (English -> French -> German -> English)

        This technique was described in:

        P. Liu, X. Wang, C. Xiang and W. Meng, "A Survey of Text Data Augmentation," 2020 International Conference on
        Computer Communication and Network Security (CCNS), 2020, pp. 191-195, doi: 10.1109/CCNS50731.2020.00049.

        :param sentence: string in english
        :return: string in english
        """
        if type(sentence) != str:
            return None

        sentence = sentence[:]
        augmented_sentence = back_translation(sentence)
        return augmented_sentence
