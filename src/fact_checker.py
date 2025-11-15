"""
Fact Checker Module - NER + Wikipedia Verification
Verifies factual claims in news articles by:
1. Extracting named entities (organizations, locations, dates)
2. Cross-referencing with Wikipedia
3. Detecting unrealistic numerical claims
"""

import spacy
import wikipediaapi
import re
from typing import Dict, List, Tuple

class FactChecker:
    def __init__(self):
        """Initialize spaCy and Wikipedia API"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='FakeNewsDetector/1.0 (Educational Project)'
        )
        
        # Suspicious patterns for quick checks
        self.numerical_checks = {
            'distance': (r'(\d+)\s*(km|kilometer|kilometres)', 500),  # Max realistic for metro/local
            'speed': (r'(\d+)\s*(km/h|kmph|mph)', 350),  # Max realistic for regular transport
            'percentage': (r'(\d+)%', 100),  # Can't exceed 100%
        }
        
        # Scam/fake news indicators
        self.scam_patterns = [
            r'forward\s+this',
            r'share\s+(urgently|immediately|now)',
            r'before\s+it.?s\s+(deleted|removed|too\s+late)',
            r'(whatsapp|facebook|google)\s+will\s+charge',
            r'send\s+to\s+\d+\s+(people|contacts|friends)',
            r'turn\s+(blue|green|red)',
            r'don.?t\s+ignore',
            r'only\s+\d+\s+(hours?|minutes?|days?)\s+left',
            r'urgent(ly)?.*message',
            r'breaking.*!\s*',
            r'shocking.*!',
            r'click\s+here\s+(before|now)',
            r'register\s+(now|immediately).*expire',
            r'limited\s+time\s+offer',
            r'act\s+(fast|now|quickly)',
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        Returns: {
            'organizations': [...],
            'locations': [...],
            'dates': [...],
            'infrastructure': [...]
        }
        """
        doc = self.nlp(text)
        
        entities = {
            'organizations': [],
            'locations': [],
            'dates': [],
            'infrastructure': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'FAC':
                entities['infrastructure'].append(ent.text)
        
        return entities
    
    def check_numerical_claims(self, text: str) -> List[Dict]:
        """Check for unrealistic numerical claims"""
        issues = []
        
        for check_type, (pattern, threshold) in self.numerical_checks.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                value = int(match[0]) if isinstance(match, tuple) else int(match)
                unit = match[1] if isinstance(match, tuple) else ''
                
                if check_type == 'percentage' and (value > threshold or value < 0):
                    issues.append({
                        'type': 'invalid_percentage',
                        'value': f"{value}%",
                        'reason': f'Invalid percentage: {value}%'
                    })
                elif value > threshold:
                    issues.append({
                        'type': f'unrealistic_{check_type}',
                        'value': f"{value}{unit}",
                        'reason': f'Unrealistic {check_type}: {value}{unit} (threshold: {threshold})'
                    })
        
        return issues
    
    def check_scam_patterns(self, text: str) -> List[Dict]:
        """Detect common scam/fake news patterns"""
        scam_issues = []
        text_lower = text.lower()
        
        for pattern in self.scam_patterns:
            if re.search(pattern, text_lower):
                scam_issues.append({
                    'type': 'scam_pattern',
                    'pattern': pattern,
                    'reason': 'Contains typical scam/chain message language'
                })
        
        # Check for excessive ALL-CAPS usage (common in scams)
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 3]
        caps_ratio = len(caps_words) / len(words) if words else 0
        
        if caps_ratio > 0.3:  # More than 30% of words are ALL-CAPS
            scam_issues.append({
                'type': 'excessive_caps',
                'pattern': 'ALL-CAPS',
                'reason': f'Excessive use of capital letters ({int(caps_ratio*100)}% of text)'
            })
        
        return scam_issues
    
    def verify_on_wikipedia(self, entity: str, context_entities: List[str]) -> Tuple[bool, str]:
        """
        Verify if entity exists on Wikipedia and check if context matches
        Returns: (verified: bool, message: str)
        """
        try:
            # Clean entity name - remove articles (The, A, An)
            entity_clean = entity.strip()
            entity_clean = re.sub(r'^(The|A|An)\s+', '', entity_clean, flags=re.IGNORECASE)
            
            # Try original entity first, then cleaned version
            page = self.wiki.page(entity)
            if not page.exists():
                page = self.wiki.page(entity_clean)
            
            if not page.exists():
                # Not finding on Wikipedia doesn't mean it's fake - could be new/regional entity
                return None, f"Could not verify '{entity}' on Wikipedia (may be legitimate but not documented)"
            
            # Check if context entities are mentioned in the page
            page_text = page.text.lower()
            found_context = []
            
            for ctx in context_entities:
                if ctx.lower() in page_text:
                    found_context.append(ctx)
            
            # Don't fail if context not found - entity might be real but context unrelated
            if context_entities and not found_context:
                return None, f"'{entity}' found on Wikipedia but couldn't verify context"
            
            return True, f"'{entity}' verified on Wikipedia"
            
        except Exception as e:
            # Errors shouldn't count as fake - network issues, etc.
            return None, f"Could not check '{entity}' (Wikipedia unavailable)"
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete fact-checking analysis
        Returns: {
            'entities': dict,
            'numerical_issues': list,
            'scam_issues': list,
            'verification_results': list,
            'confidence_adjustment': int,
            'warnings': list
        }
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Check numerical claims
        numerical_issues = self.check_numerical_claims(text)
        
        # Check for scam patterns
        scam_issues = self.check_scam_patterns(text)
        
        # Verify key entities on Wikipedia
        verification_results = []
        warnings = []
        
        # Verify organizations (like Delhi Metro, ISRO)
        for org in entities['organizations'][:2]:  # Check first 2 to avoid too many requests
            locations = entities['locations'][:3]
            verified, message = self.verify_on_wikipedia(org, locations)
            
            verification_results.append({
                'entity': org,
                'verified': verified,
                'message': message
            })
            
            # Only add warning if definitely FALSE (not None/uncertain)
            if verified == False:
                warnings.append(f"⚠️ {message}")
        
        # Calculate confidence adjustment
        confidence_penalty = 0
        
        # Each numerical issue reduces confidence by 10%
        confidence_penalty += len(numerical_issues) * 10
        
        # Each scam pattern detected increases penalty by 15%
        confidence_penalty += len(scam_issues) * 15
        
        # Only penalize for DEFINITELY failed verifications (False, not None)
        failed_verifications = sum(1 for v in verification_results if v['verified'] == False)
        confidence_penalty += failed_verifications * 15
        
        # Add numerical issue warnings
        for issue in numerical_issues:
            warnings.append(f"⚠️ {issue['reason']}")
        
        # Add scam pattern warnings
        if scam_issues:
            scam_count = len(scam_issues)
            if scam_count >= 3:
                warnings.append(f"⚠️ SCAM ALERT: Contains {scam_count} typical chain message/scam patterns")
            else:
                warnings.append(f"⚠️ Contains suspicious forwarding/urgency language")
        
        return {
            'entities': entities,
            'numerical_issues': numerical_issues,
            'scam_issues': scam_issues,
            'verification_results': verification_results,
            'confidence_adjustment': min(confidence_penalty, 50),  # Cap at 50%
            'warnings': warnings[:5]  # Show max 5 warnings
        }
    
    def get_verdict(self, ml_prediction: bool, ml_confidence: float, fact_check_result: Dict) -> Dict:
        """
        Combine ML prediction with fact-checking results
        ml_prediction: True if FAKE, False if REAL
        
        FACT-CHECKER OVERRIDE LOGIC:
        - If numerical/factual issues found, override ML to FAKE NEWS
        - Fact-checking takes priority over ML writing style analysis
        """
        has_issues = len(fact_check_result['warnings']) > 0
        confidence_penalty = fact_check_result['confidence_adjustment']
        
        # STRONG OVERRIDE: If we found ANY factual issues, classify as FAKE
        if has_issues:
            # Calculate fake confidence based on severity
            if confidence_penalty >= 20:
                # Serious issues (multiple problems or major discrepancy)
                fake_confidence = 85
                reason = '⚠️ FACT-CHECK OVERRIDE: Detected verifiable false claims that contradict reality'
            elif confidence_penalty >= 10:
                # Moderate issues (single major problem)
                fake_confidence = 75
                reason = '⚠️ FACT-CHECK ALERT: Found suspicious factual claims'
            else:
                # Minor concerns
                fake_confidence = 65
                reason = 'Potential factual inconsistencies detected'
            
            return {
                'verdict': 'FAKE NEWS',
                'adjusted_confidence': fake_confidence,
                'reason': reason
            }
        
        # No factual issues found - trust ML model
        if ml_prediction:
            # ML says FAKE, no factual issues to verify - trust ML
            return {
                'verdict': 'FAKE NEWS',
                'adjusted_confidence': ml_confidence,
                'reason': 'ML models detected fake news patterns (writing style, sensationalism)'
            }
        else:
            # ML says REAL, no factual issues found - trust ML
            return {
                'verdict': 'REAL NEWS',
                'adjusted_confidence': ml_confidence,
                'reason': 'No factual inconsistencies detected, writing style appears legitimate'
            }
