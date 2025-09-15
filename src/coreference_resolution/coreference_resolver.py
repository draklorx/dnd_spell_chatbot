from .chat_context import ChatContext
import re
from typing import Optional, Dict, List

class CoreferenceResolver:
    def __init__(self, chat_context: ChatContext):
        self.chat_context = chat_context
        
        # Define pronoun patterns for different entity types
        self.pronoun_patterns = {
            "SPELL": [
                r'\bit\b',
                r'\bits\b', 
                r'\bthat\b',
                r'\bthis\b',
                r'\bthe spell\b',
                r'\bthat spell\b',
                r'\bthis spell\b',
                r'\bthe one\b',
                r'\bthat one\b'
            ],
            "SCHOOL": [
                r'\bthat school\b',
                r'\bthis school\b',
                r'\bthe school\b',
                r'\bit\b',
                r'\bthat\b'
            ],
            "DAMAGE_TYPE": [
                r'\bthat damage type\b',
                r'\bthis damage type\b', 
                r'\bthe damage type\b',
                r'\bthat type\b',
                r'\bthis type\b',
                r'\bit\b',
                r'\bthat\b'
            ],
            "CLASS": [
                r'\bthat class\b',
                r'\bthis class\b',
                r'\bthe class\b',
                r'\bit\b',
                r'\bthat\b'
            ],
            "LEVEL": [
                r'\bthat level\b',
                r'\bthis level\b',
                r'\bthe level\b',
                r'\bit\b',
                r'\bthat\b'
            ]
        }
        
        # Priority order for resolving ambiguous pronouns
        self.resolution_priority = ["SPELL", "DAMAGE_TYPE", "SCHOOL", "CLASS", "LEVEL"]

    def resolve_coreferences(self, message: str) -> Dict[str, str]:
        """
        Resolve coreferences in the message and return a mapping of entity types to values.
        
        Args:
            message: The user's input message
            
        Returns:
            Dict mapping entity labels to resolved values
        """
        message_lower = message.lower()
        resolved_entities = {}
        
        # Check for specific entity type pronouns first (most specific)
        for entity_type, patterns in self.pronoun_patterns.items():
            if entity_type == "SPELL":
                continue  # Handle spell pronouns separately due to their generic nature
                
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    resolved_value = self._resolve_entity_type(entity_type)
                    if resolved_value:
                        resolved_entities[entity_type] = resolved_value
                        break
        
        # Handle generic pronouns that could refer to spells
        spell_patterns = self.pronoun_patterns["SPELL"]
        generic_patterns = [r'\bit\b', r'\bits\b', r'\bthat\b', r'\bthis\b']
        
        for pattern in spell_patterns:
            if re.search(pattern, message_lower):
                # Check if this is a spell-specific pronoun or generic
                is_generic = any(re.fullmatch(p, pattern) for p in generic_patterns)
                
                if not is_generic:
                    # Spell-specific pronoun
                    resolved_value = self._resolve_entity_type("SPELL")
                    if resolved_value:
                        resolved_entities["SPELL"] = resolved_value
                        break
                else:
                    # Generic pronoun - resolve based on context and priority
                    resolved_entity = self._resolve_generic_pronoun(message_lower, pattern)
                    if resolved_entity:
                        resolved_entities.update(resolved_entity)
                        break
        
        return resolved_entities

    def _resolve_entity_type(self, entity_type: str) -> Optional[str]:
        """Resolve a specific entity type from chat context."""
        context_entity = self.chat_context.get_context(entity_type)
        return context_entity.value if context_entity else None

    def _resolve_generic_pronoun(self, message: str, pattern: str) -> Optional[Dict[str, str]]:
        """
        Resolve a generic pronoun (like 'it', 'that', 'this') based on context and priority.
        
        Args:
            message: The user's input message
            pattern: The regex pattern that matched the pronoun
            
        Returns:
            Optional dictionary with resolved entity values
        """
        resolved_entities = {}
        
        # Check context for possible resolutions
        for entity_type in self.resolution_priority:
            if entity_type == "SPELL":
                continue  # Skip SPELL here, as it's handled separately
                
            # Get the last mentioned entity of this type from context
            entity_value = self._resolve_entity_type(entity_type)
            if entity_value:
                resolved_entities[entity_type] = entity_value
                break  # Stop at the first resolved entity (highest priority)
        
        return resolved_entities if resolved_entities else None
