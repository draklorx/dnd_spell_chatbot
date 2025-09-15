import json
import nltk
from ordinal import ordinal

class DataProcessor:
    def __init__(self, raw_spell_data_path, raw_entity_data_path, processed_spell_data_path, processed_entity_data_path):

        with open(raw_spell_data_path, "r", encoding="utf-8") as f:
            self.spell_data = json.load(f)
        with open(raw_entity_data_path, "r", encoding="utf-8") as f:
            self.entity_data = json.load(f)
            
        self.processed_spell_data_path = processed_spell_data_path
        self.processed_entity_data_path = processed_entity_data_path
        
    # Pulled raw data from JSON files
    # Need to process it before use in entity recognition and data extraction
    def process_data(self):
        """Process both spell and entity data."""
        # Order matters here. If we process entity data first we'll change damage types and they won't match in spells
        self.process_spell_data()
        self.process_entity_data()

    def process_spell_data(self):
        """Process raw spell data and extract the damage types from descriptions"""
        # Get a list of damage types from entity data
        damage_types = []
        for entity in self.entity_data.get("entities", []):
            if entity["label"] == "DAMAGE_TYPE":
                damage_types.extend(entity["patterns"])

        # Process each spell description to find sentences mentioning damage types
        for spell in self.spell_data["spells"]:
            description = spell.get("description", "")
            sentences = nltk.sent_tokenize(description)
            for sentence in sentences:
                sentence_words = [word.lower() for word in nltk.word_tokenize(sentence)]
                # Ignore sentences that mention resistances, immunities, or vulnerabilities
                if not any(word in sentence_words for word in ["resistance", "immunity", "vulnerability"]):
                    for damage_type in damage_types:
                        if damage_type.lower() in sentence_words:
                            # Found a damage type mention, add to processed data
                            # Ensure damageTypes is a list
                            if not spell.get("damageTypes"):
                                spell["damageTypes"] = []
                            spell["damageTypes"].append(damage_type)
            if spell.get("damageTypes"):
                # Remove duplicates
                spell["damageTypes"] = list(set(spell["damageTypes"]))
        with open(self.processed_spell_data_path, "w", encoding="utf-8") as f:
            json.dump(self.spell_data, f, indent=2)

    def process_entity_data(self):
        """Add all spells to the entity data for the SPELL entity"""

        spell_entity = {
            "label": "SPELL",
            "patterns": []
        }

        for spell in self.spell_data["spells"]:
            spell_entity["patterns"].append(spell["name"].lower())

        self.entity_data["entities"].append(spell_entity)

        # Add level entities
        patterns = ["cantrip", "cantrips"]
        for level in range(1, 10):
            patterns.append(f"{ordinal(level)} level")
            patterns.append(f"level {level}")
        self.entity_data["entities"].append({
            "label": "LEVEL",
            "patterns": patterns
        })

        for entity in self.entity_data["entities"]:
            # add saving throw to the end of each saving throw pattern
            if entity["label"] == "SAVING_THROW":
                new_patterns = []
                for pattern in entity["patterns"]:
                    new_patterns.append(f"{pattern} saving throw")
                entity["patterns"] = new_patterns

        with open(self.processed_entity_data_path, "w", encoding="utf-8") as f:
            json.dump(self.entity_data, f, indent=2)

