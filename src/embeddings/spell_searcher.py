from spell_embedder import SpellEmbedder
import re

class SpellSearcher:
    def __init__(self, db_path="spells.db"):
        """Initialize the spell searcher."""
        self.embedder = SpellEmbedder(db_path)
    
    def _calculate_keyword_boost(self, query, sentence):
        """Calculate keyword relevance boost."""
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())
        
        # Exact word matches
        # exact_matches = len(query_words.intersection(sentence_words))
        
        # # Special keyword patterns
        # boost_score = exact_matches * 0.1

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
        
        return min(boost_score, 0.5)  # Cap boost at 0.5

    def search(self, query, specific_spell=None, min_score=0.5, max_results=5):
        """
        Search for information in spells and return ordered results.
        
        Args:
            query: User's question
            specific_spell: Optional spell name to search within
            top_k: Number of results to return
        
        Returns:
            Ordered text response
        """
        results = self.embedder.search_spells(query, specific_spell, 25) # get a bunch we'll filter them here

        # Apply keyword boosting
        boosted_results = []
        failover_results = []
        for result in results:
            text, field, order, similarity = result[:4]
            keyword_boost = self._calculate_keyword_boost(query, text)
            boosted_similarity = similarity + keyword_boost

            boosted_result = list(result)
            boosted_result[3] = boosted_similarity

            if boosted_similarity >= min_score:
                boosted_results.append(tuple(boosted_result))
            else:
                failover_results.append(tuple(boosted_result))

        if len(boosted_results) == 0:
            # If no boosted results meet the threshold, use failover results
            boosted_results.extend(failover_results)
            # only return half the max requested
            max_results = max_results // 2 if max_results >= 2 else 1

        # Re-sort and take top_k
        boosted_results.sort(key=lambda x: x[3], reverse=True)

        # clear duplicates
        seen = set()
        unique_results = []
        for result in boosted_results:
            if result[0] not in seen:
                seen.add(result[0])
                unique_results.append(result)

        results = unique_results[:max_results]

        print("RESULTS:")
        for (text, field, order, score) in results:
            print(f"{order}. {text} (score: {score:.3f})")

        if not results:
            return "No relevant information found."
        
        if specific_spell or (results and len(results[0]) == 4):
            # Single spell search - order by sentence order
            spell_name = specific_spell
            if not spell_name and len(results[0]) == 5:
                spell_name = results[0][4]
            
            # Get the relevant sentences and sort by order
            relevant_sentences = []
            for result in results:
                if len(result) == 5:
                    text, field, order, score, name = result
                    relevant_sentences.append((text, field, order, score))
                else:
                    text, field, order, score = result
                    relevant_sentences.append((text, field, order, score))
            
            # Sort by source field priority and sentence order
            field_priority = {'description': 0, 'higherLevelSlot': 1, 'cantripUpgrade': 2}
            relevant_sentences.sort(key=lambda x: (field_priority.get(x[1], 999), x[2]))
            
            # Combine sentences
            response_parts = [sentence[0] for sentence in relevant_sentences]
            response = " ".join(response_parts)
            
            if spell_name:
                return f"**{spell_name}**: {response}"
            else:
                return response
        
        else:
            # Cross-spell search - show spell names
            response_parts = []
            for result in results:
                text, field, order, score, spell_name = result
                response_parts.append(f"**{spell_name}**: {text}")
            
            return "\n\n".join(response_parts)
    
    def close(self):
        """Close the searcher."""
        self.embedder.close()

def main():
    """Interactive search interface."""
    searcher = SpellSearcher()
    
    print("D&D Spell Search System")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            response = searcher.search(query, min_score=0.5, max_results=5)
            print(f"\nAccording to the spell: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    searcher.close()
    print("Goodbye!")

if __name__ == "__main__":
    main()