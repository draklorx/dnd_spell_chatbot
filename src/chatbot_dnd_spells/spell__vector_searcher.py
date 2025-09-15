from embeddings import VectorSearcher
import re
from utils.colors import YELLOW, RESET

class SpellVectorSearcher(VectorSearcher):
    def __init__(self, db_path):
        """Initialize the spell searcher."""
        super().__init__(db_path)
        self.debug = False
    
    def _calculate_keyword_boost(self, query, sentence):
        """Calculate keyword relevance boost."""
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())
        
        boost_score = 0.0

        # Boost for damage + die roll pattern e.g. 8d6 or 1d10
        if 'damage' in query.lower() and re.search(r'\b\d+d\d+\b', sentence):
            boost_score += 0.2
        
        # Boost for quantity questions and numbers
        # Only boost if a standalone number is present, but NOT if "damage" appears exactly two words after the number
        if any(word in query.lower() for word in ['how many', 'number']):
            # Match a number not followed by a word and then "damage" ie Force damage or Fire damage
            # and NOT if the number is preceded by "level above" (e.g., "level above 5")
            number_pattern = r'\b(?<!level above\s)(\d+)\b(?!\s+\w+\s+damage\b)'
            word_number_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b'
            # Check for digit numbers not preceded by "level"/"levels" and not followed by "damage"
            if re.search(number_pattern, sentence.lower()) or \
                (re.search(word_number_pattern, sentence.lower()) and not re.search(r'(level|levels)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b', sentence.lower())):
                boost_score += 0.2
        
        if any(word in query.lower() for word in ['save', 'saving throw']) and 'saving throw' in sentence.lower():
            boost_score += 0.2

        if any(word in query.lower() for word in ['aoe', 'area of effect', 'radius', 'area', 'diameter']) and \
            any(word in sentence.lower() for word in ['radius', 'area of effect', 'sphere', 'cylinder', 'cone', 'cube', 'line', 'diameter']):
            boost_score += 0.2
        return boost_score

    def search(self, query, spell_name, rec_score=0.5, min_score=0.4, max_results=5):
        """
        Search for information in spells and return ordered results.
        
        Args:
            query: User's question
            min_score: Minimum similarity score to consider
            max_results: Maximum number of results to return
        
        Returns:
            Ordered text response
        """
        results = super().search(query, spell_name, 25) # get a bunch we'll filter them here

        # Apply keyword boosting
        boosted_results = []
        failover_results = []
        for result in results:
            keyword_boost = self._calculate_keyword_boost(query, result.chunk_context)
            boosted_similarity = result.similarity_score + keyword_boost

            result.similarity_score = boosted_similarity

            if boosted_similarity >= rec_score:
                boosted_results.append(result)
            elif boosted_similarity >= min_score:
                failover_results.append(result)

        if len(boosted_results) == 0:
            # If no boosted results meet the threshold, use failover results
            boosted_results.extend(failover_results)

        # Re-sort and take top_k
        boosted_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # clear duplicates
        seen = set()
        unique_results = []
        for result in boosted_results:
            if result.chunk_context not in seen:
                seen.add(result.chunk_context)
                unique_results.append(result)

        results = unique_results[:max_results]

        if self.debug:
            print(f"{YELLOW}Debug: Search results for query for {spell_name}{RESET}")
            print(f"{YELLOW}Minimum score: {min_score} | Recommended score: {rec_score}{RESET}")
            # print results in score order
            debug_sorted = sorted(results, key=lambda x: x.similarity_score, reverse=True)
            for result in debug_sorted:
                print(f"{YELLOW}score: {result.similarity_score:.3f} - {result.chunk_context}{RESET}")

        if not results:
            return "No relevant information found."
        
        # Sort by source original sentence order
        results.sort(key=lambda x: (x.position))
        
        # Combine sentences
        response_parts = [result.chunk_context for result in results]
        response = " ".join(response_parts)

        return f"According to {spell_name}: {response}"
