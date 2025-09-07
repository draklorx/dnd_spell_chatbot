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
        if any(word in query.lower() for word in ['how many', 'number']) and (re.search(r'\b\d+\b', sentence) or re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', sentence.lower())):
            boost_score += 0.2
        
        return min(boost_score, 0.5)  # Cap boost at 0.5

    def search(self, query, specific_spell=None, top_k=3):
        """
        Search for information in spells and return ordered results.
        
        Args:
            query: User's question
            specific_spell: Optional spell name to search within
            top_k: Number of results to return
        
        Returns:
            Ordered text response
        """
        results = self.embedder.search_spells(query, specific_spell, top_k * 2)
        
        # Apply keyword boosting
        boosted_results = []
        for result in results:
            text, field, order, similarity = result[:4]
            keyword_boost = self._calculate_keyword_boost(query, text)
            boosted_similarity = similarity + keyword_boost
            
            boosted_result = list(result)
            boosted_result[3] = boosted_similarity
            boosted_results.append(tuple(boosted_result))
        
        # Re-sort and take top_k
        boosted_results.sort(key=lambda x: x[3], reverse=True)
        results = boosted_results[:top_k]
        
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
            response = searcher.search(query, top_k=3)
            print(f"\nAccording to the spell: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    searcher.close()
    print("Goodbye!")

if __name__ == "__main__":
    main()