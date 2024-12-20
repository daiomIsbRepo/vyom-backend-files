import json
import os
import openai
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict
from enum import Enum
import re

class ToolType(Enum):
    EVALUATOR = 'Evaluator'
    ENHANCER = 'Enhancer'
    WRITER = 'Writer'

class ValidationError(Exception):
    pass

class ResponseParser:
    @staticmethod
    def _clean_content(content: str) -> str:
        """Clean and format the content response while preserving line breaks"""
        # Split into lines and remove empty lines
        lines = [line.strip() for line in content.split('\n')]
        
        # Remove common prefixes/markers while preserving structure
        skip_markers = [
            'enhanced copy:',
            'generated copy:',
            'here\'s the',
            'output:',
            '---begin',
            '---end'
        ]
        
        # Filter out lines with skip markers
        content_lines = []
        skip_next = False
        for line in lines:
            if not line:
                continue
            
            lower_line = line.lower()
            if any(marker in lower_line for marker in skip_markers):
                skip_next = True
                continue
            
            if skip_next:
                skip_next = False
                if any(marker in lower_line for marker in skip_markers):
                    continue
            
            content_lines.append(line)

        # Join lines with newline character to preserve formatting
        clean_content = '\n'.join(content_lines)
        
        # Remove any prefix markers from the first line
        for prefix in ['enhanced copy:', 'generated copy:', 'here\'s the', 'output:']:
            if clean_content.lower().startswith(prefix):
                clean_content = clean_content[len(prefix):].strip()
        
        return clean_content.strip()

    @staticmethod
    def parse_evaluator_response(content: str) -> Dict[str, str]:
        """Parse Evaluator tool response with improved robustness"""
        sections = {
            'response': '',  # Maps to GENERAL_FEEDBACK
            'rating': '',    # Maps to RATING
            'mistakes': '',  # Maps to MISTAKES
            'rulesViolated': '' # Maps to RULES_VIOLATED
        }
        
        # More flexible regex pattern that handles various formatting
        section_patterns = {
            'response': r'(?:^|\n)\s*[-*]*\s*\*?\*?GENERAL[_\s]?FEEDBACK\*?\*?:?\s*',
            'rating': r'(?:^|\n)\s*[-*]*\s*\*?\*?RATING\*?\*?:?\s*',
            'mistakes': r'(?:^|\n)\s*[-*]*\s*\*?\*?MISTAKES\*?\*?:?\s*',
            'rulesViolated': r'(?:^|\n)\s*[-*]*\s*\*?\*?RULES[_\s]?VIOLATED\*?\*?:?\s*'
        }

        # Find all section positions
        section_positions = []
        for section, pattern in section_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                section_positions.append((match.start(), section))
        
        # Sort positions to process sections in order
        section_positions.sort()
        
        # Extract content between sections
        for i, (pos, section) in enumerate(section_positions):
            # Get the content until the next section or end of string
            start = pos + len(re.search(section_patterns[section], content[pos:], re.IGNORECASE).group(0))
            end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(content)
            
            # Extract and clean the section content while preserving line breaks
            section_content = content[start:end].strip()
            
            # Clean up formatting characters but preserve line breaks
            cleaned_lines = []
            for line in section_content.split('\n'):
                line = line.strip()
                if line:
                    # Remove bullet points and asterisks from the start of the line
                    line = re.sub(r'^\s*[-*]\s*', '', line)
                    line = re.sub(r'\*\*', '', line)
                    cleaned_lines.append(line)
            
            sections[section] = '\n'.join(cleaned_lines)

        return sections

    @staticmethod
    def extract_sections(content: str) -> Dict[str, str]:
        """Extract sections with fallback mechanisms while preserving line breaks"""
        sections = ResponseParser.parse_evaluator_response(content)
        
        if not ResponseParser.validate_response(sections):
            simple_sections = {}
            current_section = None
            current_lines = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                lower_line = line.lower()
                if 'general_feedback' in lower_line or 'feedback' in lower_line:
                    if current_section and current_lines:
                        simple_sections[current_section] = '\n'.join(current_lines)
                    current_section = 'response'
                    current_lines = []
                elif 'rating' in lower_line:
                    if current_section and current_lines:
                        simple_sections[current_section] = '\n'.join(current_lines)
                    current_section = 'rating'
                    current_lines = []
                elif 'mistakes' in lower_line:
                    if current_section and current_lines:
                        simple_sections[current_section] = '\n'.join(current_lines)
                    current_section = 'mistakes'
                    current_lines = []
                elif 'rules_violated' in lower_line or 'rules violated' in lower_line:
                    if current_section and current_lines:
                        simple_sections[current_section] = '\n'.join(current_lines)
                    current_section = 'rulesViolated'
                    current_lines = []
                elif current_section:
                    current_lines.append(line)
            
            # Add the last section
            if current_section and current_lines:
                simple_sections[current_section] = '\n'.join(current_lines)
            
            # Use simple sections if they're more complete
            if len(simple_sections) == 4 and ResponseParser.validate_response(simple_sections):
                sections = simple_sections

        return sections

    @staticmethod
    def validate_response(sections: Dict[str, str]) -> bool:
        """Validate that all sections have content"""
        return all(bool(content.strip()) for content in sections.values())

    @staticmethod
    def extract_sections(content: str) -> Dict[str, str]:
        """Extract sections with fallback mechanisms"""
        sections = ResponseParser.parse_evaluator_response(content)
        
        if not ResponseParser.validate_response(sections):
            # Fallback: try simpler parsing if sophisticated parsing fails
            simple_sections = {}
            current_section = None
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers in a more lenient way
                lower_line = line.lower()
                if 'general_feedback' in lower_line or 'feedback' in lower_line:
                    current_section = 'response'
                    simple_sections[current_section] = ''
                elif 'rating' in lower_line:
                    current_section = 'rating'
                    simple_sections[current_section] = ''
                elif 'mistakes' in lower_line:
                    current_section = 'mistakes'
                    simple_sections[current_section] = ''
                elif 'rules_violated' in lower_line or 'rules violated' in lower_line:
                    current_section = 'rulesViolated'
                    simple_sections[current_section] = ''
                elif current_section:
                    if simple_sections[current_section]:
                        simple_sections[current_section] += '\n'
                    simple_sections[current_section] += line
            
            # Use simple sections if they're more complete
            if len(simple_sections) == 4 and ResponseParser.validate_response(simple_sections):
                sections = simple_sections

        return sections


    @staticmethod
    def _clean_content(content: str) -> str:
        """Clean and format the content response"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        skip_markers = [
            'enhanced copy:',
            'generated copy:',
            'here\'s the',
            'output:',
            '---begin',
            '---end'
        ]
        content_lines = [line for line in lines if not any(marker in line.lower() for marker in skip_markers)]
        clean_content = ' '.join(content_lines)
        for prefix in ['enhanced copy:', 'generated copy:', 'here\'s the', 'output:']:
            if clean_content.lower().startswith(prefix):
                clean_content = clean_content[len(prefix):].strip()
        return clean_content

class PromptBuilder:
    @staticmethod
    def _clean_data(data: Dict[str, Any]) -> Dict[str, str]:
        """Clean and standardize input data"""
        return {k: str(v) if v is not None else '' for k, v in data.items()}
        
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing escape characters and extra quotes"""
        if not text:
            return ''
        text = text.strip('"')
        text = text.replace('\\n', '\n')
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text

    @staticmethod
    def _format_emoji_instruction(include_emoji: bool) -> str:
        """Format emoji instruction based on boolean value"""
        return 'Include' if include_emoji else 'Do not include'

    @staticmethod
    def _map_evaluator_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Map frontend fields to prompt fields for Evaluator"""
        mapped_data = {
            'uinput': PromptBuilder._clean_text(data.get('copyText', '') or data.get('uinput', '')),
            'crmChannel': data.get('crmChannel', ''),
            'callToAction': data.get('callToAction', '')  # Add the optional callToAction field
        }
        return mapped_data

    @staticmethod
    def _map_enhancer_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Map frontend fields to prompt fields for Enhancer"""
        mapped_data = {
            'uinput': PromptBuilder._clean_text(data.get('copyText', '') or data.get('uinput', '')),
            'crmChannel': data.get('crmChannel', ''),
            'brandVoice': data.get('brandVoice', ''),
            'improvementContext': data.get('improvementContext', ''),
            'customContext': data.get('customContext', '')
        }
        optional_fields = {
            'cta': '',
            'pointOfView': '',
            'wordLimit': '',
            'audience': '',
            'toneOfVoice': '',
            'callToAction': '',
            'readabilityIndex': '',
            'personalization': '',
            'productTitle': '',
            'vendor': '',
            'productType': '',
            'keywords': '',
            'includeEmoji': 'Do not include',
            'numberOfLines': ''
        }
        for field, default in optional_fields.items():
            if field == 'includeEmoji':
                mapped_data[field] = PromptBuilder._format_emoji_instruction(data.get(field, False))
            else:
                mapped_data[field] = data.get(field, default)
        return mapped_data

    @staticmethod
    def _map_writer_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Map frontend fields to prompt fields for Writer"""
        mapped_data = {
            'campaignDescription': data.get('campaignDescription', ''),
            'selectedCampaign': data.get('selectedCampaign', ''),
            'crmChannel': data.get('crmChannel', ''),
            'brandVoice': data.get('brandVoice', ''),
            'wordLimit': data.get('wordLimit', ''),
            'numberOfLines': data.get('numberOfLines', ''),
            'callToAction': data.get('callToAction', ''),
            'callToAction': data.get('callToAction', ''),
            'style': data.get('style', ''),
            'audience': data.get('audience', ''),
            'toneOfVoice': data.get('toneOfVoice', ''),
            'pointOfView': data.get('pointOfView', ''),
            'readabilityIndex': data.get('readabilityIndex', ''),
            'personalization': data.get('personalization', ''),
            'productTitle': data.get('productTitle', ''),
            'vendor': data.get('vendor', ''),
            'productType': data.get('productType', ''),
            'keywords': data.get('keywords', ''),
            'includeEmoji': PromptBuilder._format_emoji_instruction(data.get('includeEmoji', False)),
            'dynamicTokens': 'Include' if data.get('dynamicTokens', False) else 'Do not include',
        }
        return mapped_data

    @staticmethod
    def _validate_writer_data(data: Dict[str, Any]) -> None:
        """Validate required fields for Writer tool"""
        required_fields = {
            'campaignDescription': 'Campaign Description',
            'selectedCampaign': 'Campaign Type',
            'crmChannel': 'CRM Channel',
            'brandVoice': 'Brand Voice'
        }
        errors = [label for field, label in required_fields.items() if not data.get(field)]
        if errors:
            raise ValidationError(f"Missing required fields: {', '.join(errors)}")

    @classmethod
    def build_prompt(cls, tool: str, data: Dict[str, Any]) -> str:
        """Build prompt based on tool type, channel and input data"""
        tool_type = ToolType(tool)
        print("Input data:", json.dumps(data, indent=2))
        
        # Select template based on channel
        crm_channel = data.get('crmChannel', '').lower()
        templates = SMS_PROMPT_TEMPLATES if crm_channel == 'sms' else WHATSAPP_PROMPT_TEMPLATES
        
        # Map data based on tool type
        if tool_type == ToolType.WRITER:
            mapped_data = cls._map_writer_fields(data)
            cls._validate_writer_data(data)
        elif tool_type == ToolType.ENHANCER:
            mapped_data = cls._map_enhancer_fields(data)
        elif tool_type == ToolType.EVALUATOR:
            mapped_data = cls._map_evaluator_fields(data)
        else:
            mapped_data = data
            
        print("Mapped data:", json.dumps(mapped_data, indent=2))
        prompt = templates[tool_type.value].format_map(defaultdict(str, mapped_data))
        print("Generated prompt:", prompt)
        return prompt

def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized API response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body)
    }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler function"""
    try:
        print("Incoming request:", json.dumps(event.get('body', '{}')))
        try:
            body = json.loads(event.get('body', '{}'))
            
            # Add logging for improvementContext if present
            if 'improvementContext' in body:
                print(f"Improvement contexts requested: {body['improvementContext'].split(',')}")
                
        except json.JSONDecodeError as e:
            print("JSON Parse Error:", str(e))
            return create_response(400, {"error": "Invalid JSON in request"})

        # Validate tool type
        tool = body.get('tool', '').strip()
        if not tool or tool not in [t.value for t in ToolType]:
            return create_response(400, {"error": "Invalid tool type"})

        # Build the prompt
        try:
            prompt = PromptBuilder.build_prompt(tool, body)
        except ValidationError as e:
            print("Validation Error:", str(e))
            return create_response(400, {"error": str(e)})

        # Select the model based on crmChannel
        crm_channel = body.get('crmChannel', '').lower()
        model = "ft:gpt-4o-mini-2024-07-18:personal::AakjjmKa" if crm_channel == "sms" else "ft:gpt-4o-mini-2024-07-18:personal::AcykysKJ"
        #model = "ft:gpt-4o-mini-2024-07-18:personal::AakjjmKa" if crm_channel == "sms" else "ft:gpt-4o-mini-2024-07-18:personal::AYyjEuGd"
        #model = "ft:gpt-4o-mini-2024-07-18:personal::AZEfkjJt" if crm_channel == "sms" else "gpt-4o-mini"
        #model = "gpt-4o-mini" if crm_channel == "sms" else "gpt-4o-mini"
        print(f"Model used is {model}")

        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            if not response.get('choices') or not response['choices'][0].get('message', {}).get('content'):
                raise ValueError("Empty response from OpenAI")
            
            content = response['choices'][0]['message']['content'].strip()
            print("Extracted content:", content)

            # Process response
            if tool in [ToolType.WRITER.value, ToolType.ENHANCER.value]:
                # Check if emojis should be included
                include_emoji = False if crm_channel == 'sms' else body.get('includeEmoji', False)
                
                if not include_emoji:
                    # Remove emojis using regex pattern for emoji ranges
                    import re
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"  # dingbats
                        u"\U000024C2-\U0001F251" 
                        "]+", flags=re.UNICODE)
                    content = emoji_pattern.sub('', content)
                
                # Normalize line breaks and clean up whitespace
                clean_content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
                response_data = {'response': clean_content}
            else:  # EVALUATOR
                response_data = ResponseParser.parse_evaluator_response(content)

            print("Final response data:", json.dumps(response_data))
            return create_response(200, response_data)

        except openai.error.OpenAIError as e:
            print("OpenAI Error:", str(e))
            return create_response(500, {"error": f"OpenAI API error: {str(e)}"})

    except Exception as e:
        print("Unexpected Error:", str(e))
        return create_response(500, {"error": "Internal server error"})

# Prompt Templates
WHATSAPP_PROMPT_TEMPLATES = {
'Evaluator': """
Evaluate the provided marketing copy at the end (INPUT) for its effectiveness on the {crmChannel} channel.
If the provided copy is too short or lacks meaningful content (fewer than 10 words or not providing a clear message), 
please give a negative rating and indicate that the content is insufficient for proper evaluation.

Using the criteria below for analysis:
1.  **Readability Index rules**: 
    Calculate exact ARI = 4.71*(characters/words) + 0.5*(words/sentences) - 21.43
    Score strictly based on:
    - 5 points: ARI 8-10 (optimal for marketing)
    - 4 points: ARI 6-7 or 11-12
    - 3 points: ARI 4-5 or 13-14
    - 2 points: ARI 2-3 or 15-16
    - 1 point: ARI < 2 or > 16
    Rules: Count only alphabetic and numeric characters. URLs, currency, time = 1 word each.

2.  **Tone rules**: 
    5 points: If the tone of the copy fits these criteria: no use of inappropriate word, no use of complex language, no use of inconsistent tone shifts, no use of non-youth-friendly language, or not missing brand voice alignment.
    4 points: Fails to meet one of these criteria: no use of inappropriate word, no use of complex language, no use of inconsistent tone shifts, no use of non-youth-friendly language, or not missing brand voice alignment.
    3 points: Fails to meet two of these criteria: no use of inappropriate word, no use of complex language, no use of inconsistent tone shifts, no use of non-youth-friendly language, or not missing brand voice alignment.
    2 points: Fails to meet three of these criteria: no use of inappropriate word, no use of complex language, no use of inconsistent tone shifts, no use of non-youth-friendly language, or not missing brand voice alignment.
    1 point: Fails to meet all of these criteria: no use of inappropriate word, no use of complex language, no use of inconsistent tone shifts, no use of non-youth-friendly language, or not missing brand voice alignment.

3.  **Word length rules**: 
    Score strictly based on:
    Award score 5: Concise (Less than 50 words)
    Award score 4:Normal (50-70 words)
    Award score 3:Long (70-90 words)
    Award score 2:Very Long: Concise (90-120 words) Or 2 URL/ Links present
    Award score 1:Not fit for Whatsapp channel: Concise (120+ words) Or 2+ URL/ Links present

4.  **Image/Size Rules**:
    Score based on:
    - 5 points: 1 relevant image/video <5MB
    - 4 points: 1 relevant image/video, size unspecified
    - 3 points: 2 media elements
    - 2 points: 3+ media elements
    - 1 point: No media or oversized media

5.  **Personalized depth rules**:
    Start with 5 points, award:
    - 2 point: Has recipient name/placeholder after first line
    - 1 point: Has recipient name/placeholder on the first line
    - 1 point: References past purchase
    - 1 point: Location-specific content
    - 1 point: Personalized offer
    Divide total by maximum possible (5)

6.  **Call to Action (CTA) rules**:
    Score based on:
    Condition 1: Message body has more than 2 URL links or Data in Call to Action field is #NA .Deduct 4 Points 
    Condition 2 (Check this condition Only if condition 1 not satisfied): Check only the Data passed in field: {callToAction} and Deduct 1 point from base score 5 for each of the following violations
    URL/Link/CTA is not present 
    URL/Link/CTA is overly verbose or not concise or not branded.
    URL/Link/CTA is irrelevant to the offer or contains more than 1 URL/Link/CTA, causing confusion.
    URL/Link/CTA is irrelevant to the offer or contains more than 2 URL/Link/CTA.

7.  **Emoji rules**:
    Score based on:
    - 5 points: 1-3 relevant emojis
    - 4 points: 4-5 relevant emojis
    - 3 points: 6-7 emojis
    - 2 points: 8+ emojis
    - 1 point: Irrelevant or no emojis when needed

8.  Engagement Appeal rules**:
    Start with 5 points, award:
    - 2 points: Clear FOMO/urgency element
    - 1 point: Trend/moment reference
    - 1 point: Strong "Preview" text (opening) should be compelling 
    - 1 point: Clear value proposition
    Divide total by maximum possible (5)

9.  **Brand Alignment rules**:
    Start with 5 points, award:
    - 1 point: Effective brand name placement
    - 1 point: Consistent brand voice
    - 1 point: Brand-specific offers/discounts
    - 1 point: Brand value proposition
    - 1 point: Brand-specific keywords
    Divide total by maximum possible (5)

10. **Compliance Consideration rules**:
    Start with 5 points, deduct 1 point each for:
    - Derogatory language
    - Non-compliant personal data use
    - Inappropriate content
    - Misleading claims
    - Policy violations
    
Calculate Final Score as follows:
8.75%*(Readability score) + 8.75%*(Tone score) + 11.25%*(Word length score) + 8.75%*(Image/Size score) + 18.75%*(Personalization score) + 18.5%*(CTA score) + 6.25%*(Emoji score) + 6.25%*(Engagement Appeal score) + 8.75%*(Brand Alignment score) + 3.75%*(Compliance score)

CTA to evaluate (if provided): {callToAction}

Provide the response of each aspect STRICTLY in the below format:
    
GENERAL_FEEDBACK: Summarize the result and provide rationale on following aspects: opening, body, clarity, conciseness, effectiveness, and engagement. Format the feedback creatively: Start positive feedback with a ✅ (tick emoji) followed by the point. Start negative feedback with a ❌ (minus emoji) followed by the point. Present each point on a new line for clarity. Feedback for call to action and preview text (first few words) should be present. Ensure the tone is constructive and actionable, providing specific suggestions where necessary.

RATING: Simulate a rating process over 200 iterations to calculate the Final Score using the formula:
Final Score = 8.75%(Readability score) + 8.75%(Tone score) + 11.25%(Word length score) + 8.75%(Image/Size score) + 18.75%(Personalization score) + 18.5%(CTA score) + 6.25%(Emoji score) + 6.25%(Engagement Appeal score) + 8.75%(Brand Alignment score) + 3.75%(Compliance score)Randomly assign scores on a scale of 1 to 5 for each metric during each iteration.Calculate the Final Score for each iteration using the above formula.Perform this calculation 200 times.At the end of the 200 iterations, provide only the average Final Score out of 5, rounded to one decimal place.Ensure consistency and unbiased randomness across all iterations. Do not display the formula, intermediate values, or the scores from individual iterations in the output. provide the rating in the standard format: "X/5" (e.g., 4.2/5), where X is the score out of 5.The output should only include the score in the format "X/5" without any additional explanation or text.

RULES_VIOLATED: Suggest concisely using 4-Cs (Clear, Concise, Compelling, Consistent).
Identify specific suggestions for each parameter and present the feedback in a creative format: start with violated parameters under 4-Cs (e.g., 'Clear:') with a 🚨 (emoji) followed by the issue.
Suggestion should not contradict the violated rules of GENERAL_FEEDBACK section.

Present the response as bullet points instead of paragraphs where applicable.
Use brand and channel-specific rules as applicable. Avoid adding input prompt in your response.

INPUT: Copy to evaluate is below-
{uinput}
    """,
    'Enhancer': """
You are a professional copywriter specializing in marketing optimization. Your task is to enhance the following marketing copy to achieve maximum scores on these precise evaluation criteria:

ORIGINAL TEXT:
{uinput}

For the brand {brandVoice}, enhance this copy by optimizing for these exact scoring criteria:

SCORING CRITERIA TO OPTIMIZE FOR:

1. Readability Score (8.75%):
   - Target ARI score between 8-10 for maximum points
   - ARI Formula: 4.71*(characters/words) + 0.5*(words/sentences) - 21.43
   - Count rules: URLs/currency/time = 1 word each

2. Tone Score (8.75%):
   - Must use positive language only
   - Avoid formal/complex language
   - Maintain consistent tone throughout
   - Use youth-friendly language
   - Align with brand voice

3. Word Length Score (11.25%):
   - Target 20-40 words for maximum score
   - Count: URLs=1 word, Emojis=0 words
   - Must exceed 10-word minimum

4. Image/Size Elements (8.75%):
   - Include exactly one relevant media reference
   - If video, specify size under 5MB
   - Don't overload with multiple media elements

5. Personalization Score (18.75%):
   MUST include ALL:
   - Recipient name placeholder in parentheses
   - Past purchase reference
   - Location-specific element
   - Personalized offer/recommendation

6. Call to Action Score (18.5%):
   - Include exactly one clear, specific CTA
   - Style it as {callToAction}
   - Make it relevant to the offer
   - Keep it concise and actionable

7. Emoji Usage (6.25%):
   - {includeEmoji} emojis
   - If included, use 1-3 relevant emojis maximum
   - Ensure they match brand personality

8. Engagement Appeal (6.25%):
   MUST include ALL:
   - Clear FOMO/urgency element
   - Trend/moment reference
   - Strong opening hook (first 4-5 words)
   - Clear value proposition

9. Brand Alignment (8.75%):
   MUST include ALL:
   - Strategic brand name placement
   - Brand voice consistency
   - Brand-specific offers
   - Value proposition
   - Brand keywords

10. Compliance (3.75%):
    MUST avoid ALL:
    - Derogatory language
    - Non-compliant personal data use
    - Inappropriate content
    - Misleading claims
    - Policy violations

ENHANCEMENT CONTEXT:
- Channel: {crmChannel}
- Improvement Focus: {improvementContext}
- Additional Context: {customContext}
- Point of View: {pointofView}
- Word Limit: {wordLimit}
- Audience: {audience}
- Tone: {toneOfVoice}
- Product Details: {productTitle} | {vendor} | {productType}
- Keywords: {keywords}

CRITICAL REQUIREMENTS:
1. Focus heavily on Personalization (18.75%) and CTA (18.5%) as they carry highest weights
2. Strictly optimize for 20-40 word count (highest score range)
3. Ensure ARI formula calculates to 8-10 range
4. Return ONLY the enhanced copy, no explanations
5. Remove any hashtags
6. Format specifically for {crmChannel}
7. If using personalization like addressing someone by name, it should never be in the first sentence. it should be in the third sentence if at all.
8. Use line breaks including empty lines to improve formatting. Thing should not like a paragraph on the screen

Generate the enhanced copy that will maximize scores across all these quantitative criteria while maintaining natural, engaging flow. 
Take the given input copy and enhance it so that the Final Score of the enhanced copy is always greater than or equal to the Final Score of the input copy. Use the following scoring criteria:
Final Score Calculation:
8.75%(Readability score) + 8.75%(Tone score) + 11.25%(Word length score) + 8.75%(Image/Size score) + 18.75%(Personalization score) + 18.5%(CTA score) + 6.25%(Emoji score) + 6.25%(Engagement Appeal score) + 8.75%(Brand Alignment score) + 3.75%(Compliance score)
Evaluate the input copy's score based on the above formula.
Modify and enhance the copy in a way that systematically improves its scores across the relevant criteria.
Ensure the Final Score of the enhanced copy is greater than or equal to the input copy’s Final Score.
Only provide enhanced copy in the output and no scoring details and rules or explanation

    """,
    'Writer': """
    You are a professional copywriter. 
    Generate a marketing copy for {crmChannel} following the below criteria:
    1.  **Campaign Purpose**: Describe the purpose of the campaign {campaignDescription}. Then, select the campaign {selectedCampaign}. Is the copy for Promotional or Seasonal or Holiday?
    2.  **Tone Quality**: Tailor the tone and language to the audience's characteristics: {audience}, considering their age group (young adults, professionals, etc.) and preferences style - {style}.
    For a humorous tone, incorporate playful language, wit, or light sarcasm.
    3.  **Specify Style**: Generate a marketing copy in a {style} tone. The copy should be engaging, aligned with the brand's persona, and tailored for {crmChannel} (Ensure the style reflects the following characteristics:Humorous: Light-hearted, witty, and entertaining, with subtle puns or jokes that match the audience's sensibilities.Formal: Professional, respectful, and structured, suitable for an audience seeking reliability and expertise.
          Casual: Friendly, conversational, and relaxed, creating a sense of approachability.
    4.  **Word Limit**: Strictly adhere to the {wordLimit}. Copy should be concise (0-50 words) or normal (50-100 words) or long (100-150 words). Shorter the copy better the output; it should limit the copy to 50 -100 words.
    5.  **Personalization**: Use the recipient's name and reference their past purchase (e.g., "Hi [Name], we noticed you loved [Product]!").
    6.  **Call to Action (CTA)**: Include a clear, concise CTA, like "Shop Now" or "Grab the Deal" or "link".
    7.  **Emoji Use**: {includeEmoji} emojis in the content. If included but don’t distract. Keep emojis brand-aligned if included.
    8.  **Engagement Appeal**: Use the first few words to hook the reader. Create a sense of urgency or exclusivity.
    9.  **Brand Alignment**: Highlight brand {brandVoice} name first, then discounts and the product's relevance to the audience.
    10. **Compliance Consideration**: No derogatory words, anything that could hurt sentiment of any group.

    PRODUCT DETAILS:
    - Product: Highlight {productTitle}
    - Vendor: Mention {vendor} where relevant
    - Product Type: Emphasize {productType}
    - Key Terms: Incorporate {keywords} naturally

    Write a marketing copy for {crmChannel}, strictly adhering to all these elements (especially wordlimit if requested). Ensure it is concise, clear, and visible in the WhatsApp screen.

    Generate ONLY the copy without any additional commentary or labels.Format the following text as a WhatsApp copywriting message. Ensure the formatting is attractive, with a casual and exciting tone.
    Use emojis strategically to enhance engagement and make it visually appealing. Structure the message with proper line breaks to ensure readability.
    Highlight important phrases using uppercase, symbols, or spacing for emphasis. Remove hashtag (#) from the copy, like #Shades #Lenskart #Fashion (if any)
    Generate a Call to Action (CTA) and ensure it is included at the extreme end of the generated message. The CTA should be clear, concise, and aligned with the message's purpose. For example, "CTA: Shop Now!".
    The copy should flow naturally, with the CTA serving as a strong closing statement.
    Ensure that the CTA stands out and is placed at the end of the copy, with no additional text after the CTA.
    The CTA should be directly related to the goal of the message.
    Is using personalization like addressing someone by name, it should never be in the first sentence. it should be in the third sentence if at all.
    Use line breaks including empty lines to improve formatting. Thing should not like a paragraph on the screen
 
"""
}

SMS_PROMPT_TEMPLATES = {
'Evaluator': """
Evaluate the provided marketing copy at the end (INPUT) for its effectiveness on the {crmChannel} channel.
If the provided copy is too short or lacks meaningful content (fewer than 10 words or not providing a clear message), 
please give a negative rating and indicate that the content is insufficient for proper evaluation.

These are the rules for evaluation. Use the criteria below for analysis:
1.  **Readability Index rules**: 
    - **Characters**: Count only alphabetic and numeric characters. Exclude spaces, punctuation, or other non-alphanumeric symbols.
    - **Words**: Count groups of characters separated by spaces. Calculate exact ARI = 4.71*(characters/words) + 0.5*(words/sentences) - 21.43
    Score strictly based on:
    - 5 points: ARI 6-8 (optimal for SMS)
    - 4 points: ARI 4-5 or 9-10
    - 3 points: ARI 2-3 or 11-12
    - 2 points: ARI 1 or 13-14
    - 1 point: ARI < 1 or > 14

2.  **Tone rules**: 
    Start with 5 points and deduct 1 point for each:
    - Negative word/phrase used
    - Formal/complex language
    - Inconsistent tone shifts
    - Non-youth-friendly language
    - Missing brand voice alignment

3.  **Word length rules**: 
    - Calculate the word length. If link present, it should be shortened.
    Score strictly based on:
    - 5 points: 15-25 words (optimal for SMS)
    - 4 points: 26-35 words (optimal for SMS)
    - 3 points: 10-14 words (too short)
    - 2 points: 36-45 words (long)
    - 1 point: <10 or >45 words (does not fit word length requirements for sms channel)

4.  **Personalized depth rules**: 
    - Don't rely only on content but also on whom (recipient's name)
    - Address using recipient's name
    - Bring up some element of past purchase
    - Include regional specific elements
    Score based on:
    - 2 points: Has recipient name/placeholder
    - 1 point: References past purchase
    - 1 point: Location-specific content
    - 1 point: Personalized offer

5.  **Call to Action (CTA) rules**: 
    - CTA should always be present in copies
    - CTA should be concise
    - CTA should be relevant to offer
    - Too many CTA leads to confusion
    Score based on:
    - 5 points: 1 clear, specific CTA (if url present it should be branded)
    - 4 points: 2 clear, related CTAs (if url present it should be branded)
    - 3 points: 1 generic CTA (if url present it should be branded)
    - 2 points: 2+ CTAs or unclear CTA or unbranded CTA
    - 1 point: No CTA

6.  **Engagement Appeal rules**: 
    - Copy should generate FOMO for the user
    - First 4-5 words how it is related to the moment marketing/trend
    Score based on:
    - 2 points: Clear FOMO/urgency element
    - 1 point: Trend/moment reference
    - 1 point: Strong opening hook
    - 1 point: Clear value proposition

7.  **Brand Alignment rules**: 
    - Highlight the discount element first, then product how relevant it is
    - Add elements of brand that makes the brand unique
    Score based on:
    - 1 point: Effective brand name placement
    - 1 point: Consistent brand voice
    - 1 point: Brand-specific offers/discounts
    - 1 point: Brand value proposition
    - 1 point: Brand-specific keywords

8. **Compliance Consideration rules**: 
    - Copy should not have any derogatory words
    - Anything that could hurt sentiment of any group
    - If using personal information, ensure it complies with data protection regulations
    Score based on deductions:
    -1 point deducted each for:
    - Derogatory language
    - Non-compliant personal data use
    - Inappropriate content
    - Misleading claims
    - Policy violations
    
Calculate Final Score as follows:
13.13%*(Readability score) + 8.75%*(Tone score) + 11.25%*(Word length score) + 18.75%*(Personalization score) + 18.75%*(CTA score) + 12.5%*(Engagement Appeal score) + 13.13%*(Brand Alignment score) + 3.75%*(Compliance score)

CTA to evaluate (if provided): {callToAction}

If the copy is too short or lacks substantial content, please mark the score as 1/5 or below and include a note that the content is insufficient for evaluation.
    
Provide the response of each aspect STRICTLY in the below format:
    
GENERAL_FEEDBACK: Summarize the results and provide rationale on following aspects: opening, body, clarity, conciseness, effectiveness, and engagement. Format the feedback creatively: Start positive feedback with a ✅ (tick emoji) followed by the point. Start negative feedback with a ❌ (minus emoji) followed by the point. Present each point on a new line for clarity. Ensure the tone is constructive and actionable, providing specific suggestions where necessary.

RATING: Simulate a rating process over 200 iterations to calculate the Final Score using the formula:
Final Score = 13.13%*(Readability score) + 8.75%*(Tone score) + 11.25%*(Word length score) + 18.75%*(Personalization score) + 18.75%*(CTA score) + 12.5%*(Engagement Appeal score) + 13.13%*(Brand Alignment score) + 3.75%*(Compliance score). Randomly assign scores on a scale of 1 to 5 for each metric during each iteration.Calculate the Final Score for each iteration using the above formula.Perform this calculation 200 times.At the end of the 200 iterations, provide only the average Final Score out of 5, rounded to one decimal place.Ensure consistency and unbiased randomness across all iterations.Do not display the formula, intermediate values, or the scores from individual iterations in the output. provide the rating in the standard format: "X/5" (e.g., 4.2/5), where X is the score out of 5.The output should only include the score in the format "X/5" without any additional explanation or text.

RULES_VIOLATED: List out mistakes based on the 4-C copywriting parameters: Clear, Concise, Compelling, and Consistent. Identify specific mistakes for each parameter and present the feedback in a creative format: Start each mistake with a 🚨 (mistake emoji) followed by the issue. Each point should be on a new line for clarity. Ensure the tone is professional yet supportive.

Use brand and sms channel-specific rules as applicable. Avoid adding input prompt in your response.

INPUT: Copy to evaluate is below-
{uinput}
    """,
    'Enhancer': """
You are a professional copywriter specializing in marketing optimization. Your task is to enhance the following marketing copy according to the strict requirements outlined below:

ORIGINAL TEXT:
{uinput}

For the brand {brandVoice}, please focus on the following requirements to enhance the input copy.

ENHANCEMENT REQUIREMENTS:
    1. Channel: Optimize for {crmChannel} platform with these exact scoring criteria:
       - Readability (13.13%): Target ARI 6-8 for optimal SMS readability 
       - Word Length (11.25%): Aim for 15-25 words for maximum score
       - Personalization (18.75%): Must include recipient name and past purchase references
       - CTA (18.75%): One clear, specific call-to-action with shortened link
       - Engagement (12.5%): Strong hook in first 4-5 words with FOMO element
       - Brand Alignment (13.13%): Lead with brand and relevant offers
       - Compliance (3.75%): Ensure all content is appropriate and compliant

    2. Improvement Focus: Focus on improving {improvementContext}.
    3. Additional Context: Incorporate {customContext} seamlessly.
    4. Brand Voice: Maintain it for {brandVoice} consistently.
    5. CTA Requirements: Include {cta} call(s) to action prominently, with text styled as {callToAction}.

    STYLE GUIDELINES:
    - Point of View: Enhance the copy with the point of view of {pointofView}
    - Word Limit: Strictly adhere to 15-25 words for optimal SMS length
    - Target Audience: Write for {audience}
    - Tone of Voice: Ensure {toneOfVoice} tone throughout
    - Personalization: Apply {personalization} appropriately. In case personalization is applied, keep parentheses in places where the elements of personalization should go. These elements include name of the recipient, city, references to past orders or preferences, etc.

    PRODUCT DETAILS:
    - Product Title: Highlight {productTitle}
    - Vendor: Mention {vendor} where relevant
    - Product Type: Emphasize {productType}
    - Keywords: Incorporate {keywords} naturally

    SPECIAL INSTRUCTIONS:
    - Strictly no emojis in SMS copy
    - Keep the tone and style on-brand for {brandVoice}
    - Ensure the copy is optimized for SMS (readable on small screens)
    - All links must be marked for shortening
    - Focus on improving the {improvementContext} aspect while maintaining the core message

    OUTPUT FORMAT:
    Return ONLY the enhanced copy text, with no additional commentary, labels, or explanations.

    ENHANCEMENT PROCESS:
    1. Analyze the original text and provide the enhanced version targeted to SMS channel
    2. Identify areas for improvement based on the requirements
    3. Adhere strictly to all guidelines
    4. Include product details - type, vendor, keywords for AI as provided
    5. Ensure each requirement is clearly addressed
    6. Double check for clarity, coherence, and adherence to rules
    7. Make sure to keep the format of the enhanced copy clean and readable
    8. Highlight {productType}, {productTitle}, {vendor} and {keywords} as applicable

    Key Formatting Instructions:
    1. Do not repeat the same structure or content in both versions
    2. Copy should be creative, engaging, and informal, with a dynamic tone
    3. Ensure both versions adhere to the rules of tone, engagement, personalization, and the appropriate tone for the target audience
    4. The Call to Action must be clear, concise, and relevant to the offer
    5. Keep the format clean and readable on mobile screens
    6. If using personalization, keep placeholder elements in parentheses. Personalization like addressing someone by name should never be in the first sentence. It should be in the second or third sentence if at all.
    7. Use line breaks including empty lines to improve formatting. Thing should not like a paragraph on the screen

The enhanced copy must score higher than the original when evaluated against our quantitative SMS scoring criteria. Focus particularly on:
- ARI score between 6-8
- Word count between 15-25 words
- One clear CTA with shortened link
- Strong personalization elements
- Clear FOMO/urgency in opening
- Brand voice consistency
Only provide enhanced copy in the output and no scoring details and rules or explanation
    """,
    'Writer': """
    You are a professional copywriter. 
    Generate a marketing copy for {crmChannel} following the below criteria:
    1.  **Campaign Purpose**: Describe the purpose of the campaign {campaignDescription}. Then, select the campaign {selectedCampaign}. Is the copy for Promotional or Seasonal or Holiday?
    2.  **Tone Quality**: Tailor the tone and language to the audience's characteristics: {audience}, considering their age group (young adults, professionals, etc.) and preferences style - {style}.
    For a humorous tone, incorporate playful language, wit, or light sarcasm.
    3.  **Specify Style**: Generate a marketing copy in a {style} tone. The copy should be engaging, aligned with the brand's persona, and tailored for {crmChannel} (Ensure the style reflects the following characteristics:Humorous: Light-hearted, witty, and entertaining, with subtle puns or jokes that match the audience's sensibilities.Formal: Professional, respectful, and structured, suitable for an audience seeking reliability and expertise.
          Casual: Friendly, conversational, and relaxed, creating a sense of approachability.
    4.  **Word Limit**: Strictly adhere to the wordlimit. The copy must fit on a standard SMS screen.
    5.  **Personalization**: Use the recipient's name and reference their past purchase (e.g., "Hi [Name], we noticed you loved [Product]!").
    6.  **Call to Action (CTA)**: Include a clear, concise CTA, like "Shop Now" or "Grab the Deal" or "link".
    7.  **Engagement Appeal**: Use the first few words to hook the reader. Create a sense of urgency or exclusivity.
    8.  **Brand Alignment**: Highlight brand {brandVoice} name first, then discounts and the product's relevance to the audience.
    9. **Compliance Consideration**: No derogatory words, anything that could hurt sentiment of any group.

    PRODUCT DETAILS:
    - Product: Highlight {productTitle}
    - Vendor: Mention {vendor} where relevant
    - Product Type: Emphasize {productType}
    - Key Terms: Incorporate {keywords} naturally

    Write a marketing copy for {crmChannel}, strictly adhering to all these elements (especially wordlimit if requested). Ensure it is concise, clear, and visible in the sms screen.

    **Output Requirements**:
    - Keep the copy under {WordLimit} and tailored for SMS. It should be less or 30 words only.
    - Keep the tone casual, exciting, and visually engaging.
    - Use line breaks, spacing, or symbols to emphasize key points.
    - Generate only the SMS content, formatted for readability and visual appeal.
    - if using personalization like addressing someone by name, it should never be in the first sentence. it should be in the second or third sentence if at all.

    Generate ONLY the copy without any additional commentary or labels.Format the following text as a sms copywriting message. Ensure the formatting is attractive, with a casual and exciting tone.
    Write the SMS message as described, following all guidelines. Keep it concise, clear, and captivating. Remove hashtag (#) from the copy, like #Shades #Lenskart #Fashion (if any)
    Use line breaks including empty lines to improve formatting. Thing should not like a paragraph on the screen
"""
}