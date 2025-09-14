from entity_classification import EntityRuleClassifier

class SpellEntityClassifier(EntityRuleClassifier):
    def _extract_key_value(self, text, label):
        """Extract the key part from matched entities based on label type"""
        if label == "SAVING_THROW":
            # Extract the ability name (first word)
            return text.split()[0]
        elif label == "LEVEL":
            if "level" in text.lower():
                # Extract the number/ordinal
                words = text.lower().split()
                for word in words:
                    if word in ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th"]:
                        return word[0]  # Return just the number part
                    elif word in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        return word
            elif text.lower() in ["cantrip", "cantrips"]:
                return "0"  # Cantrips are level 0
        # For other labels, return the original text
        return text
    
    def predict(self, text):
        print("Predicting entities for text:", text)
        doc = self.nlp(text)
        print(doc._.coref_chains)
        print(doc._.coref_chains.resolve(doc[31])) # This would resolve 'they' in 'they loved'
        if doc is not None:
            results = []
            for ent in doc.ents:
                key_value = self._extract_key_value(ent.text, ent.label_)
                results.append((key_value, ent.label_))
                print ("RESULT", ent.text, ent.label_, key_value)
            return results
        return []