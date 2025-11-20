"""
Fact Checker Module - NER + Wikipedia Verification + Google Fact Check API
Verifies factual claims in news articles by:
1. Extracting named entities (organizations, locations, dates)
2. Cross-referencing with Wikipedia
3. Detecting unrealistic numerical claims
4. Using Google Fact Check API for additional verification
"""

import spacy
import wikipediaapi
import re
import os
import requests
from typing import Dict, List, Tuple

class FactChecker:
    def __init__(self, google_api_key=None):
        """Initialize spaCy, Wikipedia API, and optionally Google Fact Check API"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='FakeNewsDetector/1.0 (Educational Project)'
        )
        
        # Google Fact Check API configuration
        self.google_api_key = google_api_key or os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        self.use_google_api = bool(self.google_api_key)
        self.google_api_url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
        
        if self.use_google_api:
            print("✓ Google Fact Check API enabled")
        else:
            print("ℹ️ Google Fact Check API not configured (optional)")
        
        # Suspicious patterns for quick checks
        self.numerical_checks = {
            'distance': (r'(\d+)\s*(km|kilometer|kilometres)', 500),  # Max realistic for metro/local
            'speed': (r'(\d+)\s*(km/h|kmph|mph)', 350),  # Max realistic for regular transport
            'percentage': (r'(\d+)%', 100),  # Can't exceed 100%
        }
        
        # Scam/fake news indicators
        self.scam_patterns = [
            r'forward\s+this',
            r'share\s+(urgently|immediately|now|this|before)',
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
            r'big\s+pharma.*doesn.?t\s+want',
            r'doctors?\s+(don.?t\s+want|hate|hide)',
            r'this\s+(simple|one)\s+(trick|secret|remedy)',
            r'cure.*(cancer|diabetes|disease).*\d+\s+days',
            r'(ancient|hidden|secret).*cure',
            r'completely\s+cure[ds]?',
            r'no\s+need\s+for.*(treatment|medicine|surgery)',
            r'pharmaceutical.*trying\s+to\s+ban',
            r'god\s+bless',
            r'share.*everyone.*know',
        ]
        
        # Medical misinformation patterns
        self.medical_scam_patterns = [
            r'(cures?|heals?|treats?)\s+(all|any|every)',
            r'(cancer|diabetes|heart\s+disease).*cure[ds]?.*\d+\s+days',
            r'drinking\s+(hot\s+)?water.*cure',
            r'miracle\s+(cure|treatment|remedy)',
            r'doctors?\s+reveal[^\w]*',
            r'medical\s+breakthrough.*big\s+pharma',
            r'within\s+\d+\s+days.*cure[ds]?',
            r'stage\s+\d+\s+cancer.*completely\s+cure[ds]?',
            # COVID/Vaccine conspiracy theories
            r'vaccine.*contain.*microchip',
            r'vaccine.*track',
            r'bill\s+gates.*vaccine',
            r'vaccine.*alter.*DNA',
            r'vaccine.*magnetic',
            r'5G.*covid',
            r'covid.*hoax',
            r'thousands.*died.*vaccine',
        ]
        
        # Suspicious claim patterns (fake news indicators)
        self.suspicious_claim_patterns = [
            r'announced?\s+(plans?|timeline)',  # Unverified announcements
            r'(lead|senior)\s+(researcher|scientist|administrator)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # Fake names
            r'press\s+conference.*(?:yesterday|today|last\s+\w+)',  # Recent unverified events
            r'scientists?\s+(revealed?|discovered?|announced?).*(?:compelling|shocking)',
            r'discovery.*(?:changes|revolutionizes)\s+everything',
            r'accelerate.*mission.*timeline.*based\s+on',
        ]
        
        # Known factual ranges for verification
        self.factual_ranges = {
            'mars_temperature': (-125, -14),  # °C range
            'mars_water': (0, 0),  # No confirmed liquid water on surface
            'space_mission_years': (2025, 2050),  # Realistic mission planning
            'cricket_world_cup_start': 1975,  # Men's Cricket World Cup
            'womens_cricket_wc_start': 1973,  # Women's Cricket World Cup
        }
        
        # Known sports facts
        self.sports_facts = {
            'cricket_batters': ['smriti mandhana', 'virat kohli', 'rohit sharma', 'sachin tendulkar'],
            'cricket_bowlers': ['jasprit bumrah', 'mitchell starc', 'kagiso rabada'],
        }
    
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
        
        # Check general scam patterns
        for pattern in self.scam_patterns:
            if re.search(pattern, text_lower):
                scam_issues.append({
                    'type': 'scam_pattern',
                    'pattern': pattern,
                    'reason': 'Contains typical scam/chain message language',
                    'severity': 'medium'
                })
        
        # Check medical misinformation patterns (CRITICAL)
        for pattern in self.medical_scam_patterns:
            if re.search(pattern, text_lower):
                scam_issues.append({
                    'type': 'medical_scam',
                    'pattern': pattern,
                    'reason': 'Contains dangerous medical misinformation pattern',
                    'severity': 'critical'
                })
        
        # Check suspicious claim patterns (sophisticated fake news)
        for pattern in self.suspicious_claim_patterns:
            if re.search(pattern, text_lower):
                scam_issues.append({
                    'type': 'suspicious_claim',
                    'pattern': pattern,
                    'reason': 'Contains unverified claim pattern common in fake news',
                    'severity': 'medium'
                })
        
        # Check for excessive ALL-CAPS usage (common in scams)
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 3]
        caps_ratio = len(caps_words) / len(words) if words else 0
        
        if caps_ratio > 0.3:  # More than 30% of words are ALL-CAPS
            scam_issues.append({
                'type': 'excessive_caps',
                'pattern': 'ALL-CAPS',
                'reason': f'Excessive use of capital letters ({int(caps_ratio*100)}% of text)',
                'severity': 'low'
            })
        
        # Check for excessive exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 5:
            scam_issues.append({
                'type': 'excessive_exclamation',
                'pattern': '!!!',
                'reason': f'Excessive use of exclamation marks ({exclamation_count} found)',
                'severity': 'low'
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
    
    def check_google_fact_check(self, text: str) -> List[Dict]:
        """
        Query Google Fact Check API for claims in the text
        Returns list of fact-check results from verified publishers
        """
        if not self.use_google_api:
            print("🔍 Google API not enabled (no API key)")
            return []
        
        try:
            # Extract key claims (first 500 chars to avoid too long queries)
            query_text = text[:500]
            
            print(f"🌐 Calling Google Fact Check API...")
            print(f"   Query text (first 100 chars): {query_text[:100]}...")
            
            params = {
                'key': self.google_api_key,
                'query': query_text,
                'languageCode': 'en'
            }
            
            response = requests.get(self.google_api_url, params=params, timeout=5)
            
            print(f"   API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"⚠️ Google API returned status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return []
            
            data = response.json()
            print(f"   Response data: {data}")
            
            if 'claims' not in data:
                print("   ℹ️ No claims found in Google API response")
                return []
            
            print(f"   ✅ Found {len(data.get('claims', []))} claims")
            
            fact_checks = []
            for claim in data.get('claims', [])[:3]:  # Limit to top 3 results
                claim_review = claim.get('claimReview', [{}])[0]
                
                fact_check_result = {
                    'claim': claim.get('text', 'Unknown claim'),
                    'rating': claim_review.get('textualRating', 'Unknown'),
                    'publisher': claim_review.get('publisher', {}).get('name', 'Unknown'),
                    'url': claim_review.get('url', '')
                }
                fact_checks.append(fact_check_result)
                print(f"   📊 Claim: {fact_check_result['claim'][:50]}... | Rating: {fact_check_result['rating']}")
            
            return fact_checks
            
        except requests.exceptions.Timeout:
            print("⚠️ Google Fact Check API timeout")
            return []
        except Exception as e:
            print(f"⚠️ Google Fact Check API error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def check_factual_claims(self, text: str) -> List[Dict]:
        """
        Check specific factual claims against known data
        Detects fabricated statistics and false claims
        """
        factual_issues = []
        text_lower = text.lower()
        
        # Check Mars temperature claims
        mars_temp_pattern = r'mars.*temperature.*?(-?\d+)\s*°?c'
        mars_matches = re.findall(mars_temp_pattern, text_lower)
        for temp_str in mars_matches:
            try:
                temp = int(temp_str)
                min_temp, max_temp = self.factual_ranges['mars_temperature']
                if temp < min_temp or temp > max_temp:
                    factual_issues.append({
                        'type': 'false_fact',
                        'claim': f'Mars temperature {temp}°C',
                        'reason': f'Incorrect Mars temperature claim ({temp}°C is outside realistic range {min_temp} to {max_temp}°C)',
                        'severity': 'high'
                    })
            except:
                pass
        
        # Check for liquid water on Mars claims (currently NO confirmed liquid surface water)
        if 'mars' in text_lower and 'liquid water' in text_lower and 'surface' in text_lower:
            # Check if claiming discovery/flowing water
            if re.search(r'(discover|found|flowing|liquid water).*surface', text_lower):
                factual_issues.append({
                    'type': 'false_fact',
                    'claim': 'Liquid water on Mars surface',
                    'reason': 'Claims liquid water on Mars surface (no confirmed liquid water exists on surface)',
                    'severity': 'high'
                })
        
        # Check for fabricated person names in scientific contexts
        # Pattern: Dr./Prof. [FirstName LastName] where name sounds fake
        person_pattern = r'(dr\.|prof\.|doctor|professor)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        person_matches = re.findall(person_pattern, text)
        
        # If multiple fabricated-sounding names appear, flag it
        if len(person_matches) >= 2:
            factual_issues.append({
                'type': 'unverified_sources',
                'claim': 'Multiple named sources',
                'reason': f'Contains {len(person_matches)} specific person names that cannot be independently verified',
                'severity': 'medium'
            })
        
        # Check for suspiciously precise but unverifiable statistics
        # Pattern: "X liters per meter", "analyzed X images over Y years"
        precise_stats = re.findall(r'\d+(?:\.\d+)?\s+(?:liters?|images?|samples?).*?(?:per|over|spanning)\s+\d+', text_lower)
        if len(precise_stats) >= 2:
            factual_issues.append({
                'type': 'unverifiable_precision',
                'claim': 'Suspiciously precise statistics',
                'reason': 'Contains multiple overly precise statistics that are difficult to independently verify',
                'severity': 'medium'
            })
        
        return factual_issues
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete fact-checking analysis
        Returns: {
            'entities': dict,
            'numerical_issues': list,
            'scam_issues': list,
            'factual_issues': list,
            'verification_results': list,
            'confidence_adjustment': int,
            'warnings': list,
            'google_fact_checks': list
        }
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Check numerical claims
        numerical_issues = self.check_numerical_claims(text)
        
        # Check for scam patterns
        scam_issues = self.check_scam_patterns(text)
        
        # Check factual claims (NEW - catches sophisticated fake news)
        factual_issues = self.check_factual_claims(text)
        
        # Check Google Fact Check API (if enabled)
        google_fact_checks = self.check_google_fact_check(text)
        
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
        
        # CRITICAL: Medical scams get massive penalty
        critical_issues = [s for s in scam_issues if s.get('severity') == 'critical']
        confidence_penalty += len(critical_issues) * 30  # 30% penalty each
        
        # HIGH: False factual claims
        high_factual_issues = [f for f in factual_issues if f.get('severity') == 'high']
        confidence_penalty += len(high_factual_issues) * 25  # 25% penalty each
        
        # Each numerical issue reduces confidence by 10%
        confidence_penalty += len(numerical_issues) * 10
        
        # Each medium scam pattern detected increases penalty by 15%
        medium_issues = [s for s in scam_issues if s.get('severity') == 'medium']
        confidence_penalty += len(medium_issues) * 15
        
        # Medium factual issues
        medium_factual_issues = [f for f in factual_issues if f.get('severity') == 'medium']
        confidence_penalty += len(medium_factual_issues) * 12
        
        # Low severity issues - 5% each
        low_issues = [s for s in scam_issues if s.get('severity') == 'low']
        confidence_penalty += len(low_issues) * 5
        
        # Only penalize for DEFINITELY failed verifications (False, not None)
        failed_verifications = sum(1 for v in verification_results if v['verified'] == False)
        confidence_penalty += failed_verifications * 15
        
        # Add numerical issue warnings
        for issue in numerical_issues:
            warnings.append(f"⚠️ {issue['reason']}")
        
        # Add critical medical scam warnings first
        if critical_issues:
            warnings.insert(0, f"🚨 DANGER: Detected {len(critical_issues)} dangerous medical misinformation pattern(s)")
        
        # Add high-severity factual issues
        for issue in high_factual_issues:
            warnings.insert(0 if not critical_issues else 1, f"❌ FALSE CLAIM: {issue['reason']}")
        
        # Add medium factual issues with specific details
        for issue in medium_factual_issues:
            warnings.append(f"⚠️ {issue['reason']}")
        
        # Add individual scam pattern warnings with specific details
        scam_types_shown = set()
        scam_examples = []
        
        for issue in scam_issues:
            issue_type = issue.get('type', 'unknown')
            
            # Collect one example of each type
            if issue_type not in scam_types_shown:
                scam_types_shown.add(issue_type)
                
                if issue_type == 'suspicious_claim':
                    # Extract the actual pattern matched
                    pattern = issue.get('pattern', '')
                    # Try to find what text matched this pattern
                    match = re.search(pattern, text.lower())
                    if match:
                        matched_text = match.group(0)
                        scam_examples.append(f"⚠️ Unverified claim detected: \"{matched_text}\"")
                    else:
                        scam_examples.append(f"⚠️ Unverified announcement/claim pattern detected")
                        
                elif issue_type == 'scam_pattern':
                    pattern = issue.get('pattern', '')
                    match = re.search(pattern, text.lower())
                    if match:
                        matched_text = match.group(0)
                        scam_examples.append(f"⚠️ Scam language detected: \"{matched_text}\"")
                    else:
                        scam_examples.append(f"⚠️ Chain message/forwarding language detected")
                        
                elif issue_type == 'excessive_caps':
                    scam_examples.append(f"⚠️ {issue['reason']}")
                elif issue_type == 'excessive_exclamation':
                    scam_examples.append(f"⚠️ {issue['reason']}")
        
        # Add the scam examples to warnings (max 3)
        warnings.extend(scam_examples[:3])
        
        # Add Google Fact Check warnings (if any found)
        for fact_check in google_fact_checks:
            rating = fact_check.get('rating', '').lower()
            # Flag claims rated as false, misleading, or partially false
            if any(keyword in rating for keyword in ['false', 'misleading', 'incorrect', 'inaccurate', 'disputed']):
                publisher = fact_check.get('publisher', 'Fact-checker')
                claim = fact_check.get('claim', 'Unknown')[:80]  # Limit length
                warnings.insert(0, f"🌐 {publisher}: \"{claim}\" rated as {fact_check.get('rating', 'disputed')}")
                confidence_penalty += 20  # Add penalty for fact-checked false claims
        
        return {
            'entities': entities,
            'numerical_issues': numerical_issues,
            'scam_issues': scam_issues,
            'factual_issues': factual_issues,
            'verification_results': verification_results,
            'google_fact_checks': google_fact_checks,
            'google_api_enabled': self.use_google_api,
            'confidence_adjustment': min(confidence_penalty, 50),  # Cap at 50%
            'warnings': warnings[:8]  # Show max 8 warnings to include Google results
        }
    
    def get_verdict(self, ml_prediction: bool, ml_confidence: float, fact_check_result: Dict) -> Dict:
        """
        Combine ML prediction with fact-checking results
        ml_prediction: True if FAKE, False if REAL
        
        SIMPLIFIED: Just RED (FAKE) or GREEN (REAL)
        - RED: Fact-checker found issues OR ML says fake
        - GREEN: No issues found AND ML says real
        """
        warnings = fact_check_result['warnings']
        has_issues = len(warnings) > 0
        
        # If fact-checker found ANY issues, it's FAKE (override ML)
        if has_issues:
            return {
                'verdict': 'FAKE NEWS',
                'adjusted_confidence': 95,
                'color': 'red',
                'reason': '🚨 FACT-CHECKER DETECTED: ' + warnings[0] if warnings else 'Suspicious patterns detected'
            }
        
        # No fact-check issues - use ML prediction
        if ml_prediction:
            # ML says FAKE
            return {
                'verdict': 'FAKE NEWS',
                'adjusted_confidence': ml_confidence,
                'color': 'red',
                'reason': 'ML models detected fake news patterns'
            }
        else:
            # ML says REAL, no issues found
            return {
                'verdict': 'REAL NEWS',
                'adjusted_confidence': ml_confidence,
                'color': 'green',
                'reason': '✓ Verified - No suspicious patterns detected'
            }
