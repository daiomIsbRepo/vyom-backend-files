import sys
import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import openai
import time
from enum import Enum, auto

class Channel(Enum):
    WHATSAPP = "whatsapp"
    SMS = "sms"
    
    @classmethod
    def from_string(cls, value: str) -> 'Channel':
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid channel: {value}. Must be either 'whatsapp' or 'sms'")

@dataclass
class ScoreConfig:
    min_score: float = 0.0
    max_score: float = 10.0

@dataclass
class FineTuningConfig:
    model_name: str = "gpt-4o-mini-2024-07-18"
    max_retries: int = 3
    retry_delay: int = 5
    poll_interval: int = 120
    timeout: int = 24 * 60 * 60  # 24 hours in seconds
    min_examples: int = 10

class MarketingDataProcessor:
    def __init__(self, score_config: ScoreConfig = None):
        self.score_config = score_config or ScoreConfig()
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'marketing_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.dropna(subset=['Message Body', 'Sl No', 'Brand Name'])
        cleaned_df = cleaned_df.dropna(how='all')
        self.logger.info(f"Cleaned dataframe from {len(df)} to {len(cleaned_df)} valid rows")
        return cleaned_df

    def _create_training_example(self, row: pd.Series) -> Dict:
        try:
            features = {
                'brand_name': row['Brand Name'],
                'message_body': row['Message Body'],
                'sent_date': str(row['Sent Date']),
                'call_to_action': row['Call to Action'],
                'receiver_name': str(row['Receiver Name']),
                'attachment': {
                    'has_attachment': str(row['Attachment (Yes/No)']),
                    'type': str(row.get('Attachment Type', '')),
                    'size': str(row.get('Size of Attachment', ''))
                },
                'scores': {
                    'overall': round(float(row['Final Score']), 1) if pd.notna(row['Final Score']) else 0.0,
                    'readability_index': round(float(row['Readabilty Index Final Score'])) if pd.notna(row['Readabilty Index Final Score']) else 0,
                    'tone': round(float(row['Tone Final Score'])) if pd.notna(row['Tone Final Score']) else 0,
                    'word_length': round(float(row['Word Length Final Score'])) if pd.notna(row['Word Length Final Score']) else 0,
                    'image_size': round(float(row['Image/Video Size Final Score'])) if pd.notna(row['Image/Video Size Final Score']) else 0,
                    'personalization': round(float(row['Personalization Depth Final Score'])) if pd.notna(row['Personalization Depth Final Score']) else 0,
                    'cta': round(float(row['Call To Action Final Score'])) if pd.notna(row['Call To Action Final Score']) else 0,
                    'emoji': round(float(row['Emoji Final Score'])) if pd.notna(row['Emoji Final Score']) else 0,
                    'engagement': round(float(row['Engagement Appeal Final Score'])) if pd.notna(row['Engagement Appeal Final Score']) else 0,
                    'brand_alignment': round(float(row['Brand Alignment Final Score'])) if pd.notna(row['Brand Alignment Final Score']) else 0,
                    'compliance': round(float(row['Compliance Considerations LLM Score'])) if pd.notna(row['Compliance Considerations LLM Score']) else 0
                },
                'rules': {
                    'readability': str(row.get('Readability Index Rules', '')),
                    'tone': str(row.get('Tone Rules', '')),
                    'word_length': str(row.get('Word Length Rules', '')),
                    'image_size': str(row.get('Image/Video Size Rules', '')),
                    'personalization': str(row.get('Personalization Depth Rules', '')),
                    'cta': str(row.get('Call To Action Rules', '')),
                    'emoji': str(row.get('Emoji Rules', '')),
                    'engagement': str(row.get('Engagement Appeal Rules', '')),
                    'brand_alignment': str(row.get('Brand Alignment Rules', '')),
                    'compliance': str(row.get('Compliance Rules', ''))
                },
                'metrics': {
                    'word_count': int(row['Word length']) if pd.notna(row['Word length']) else 0,
                    'sentence_count': int(row['No of Sentences']) if pd.notna(row['No of Sentences']) else 0,
                    'scroll_count': int(row['Scroll Count']) if pd.notna(row['Scroll Count']) else 0
                }
            }
            
            return {
                "messages": [
                    {"role": "system", "content": self._get_system_message(features)},
                    {"role": "user", "content": self._get_human_message(features)},
                    {"role": "assistant", "content": self._get_assistant_message(features)}
                ]
            }
        except Exception as e:
            self.logger.warning(f"Error processing row: {str(e)}")
            raise

    def _get_system_message(self, features: Dict) -> str:
        return f"""You are an expert marketing copywriter for {features['brand_name']}. Your role is to maintain and enhance the brand's unique voice and identity while creating WhatsApp marketing promotions.

Evaluate and enhance WhatsApp marketing copy while adhering to the brand guidelines."""

    def _get_human_message(self, features: Dict) -> str:
        return f"""Analyze this WhatsApp marketing copy for {features['brand_name']}:

Content: {features['message_body']}

Message Details:
- Date: {features['sent_date']}
- CTA: {features['call_to_action']}
- Recipient: {features['receiver_name']}

Message Metrics:
- Word Count: {features['metrics']['word_count']}
- Sentences: {features['metrics']['sentence_count']}
- Scroll Count: {features['metrics']['scroll_count']}

Current Scores (0-10):
- Overall: {features['scores']['overall']:.1f}
- Brand Alignment: {features['scores']['brand_alignment']}
- Tone: {features['scores']['tone']}
- Personalization: {features['scores']['personalization']}
- Engagement Appeal: {features['scores']['engagement']}
- Other Metrics:
  - Readability: {features['scores']['readability_index']}
  - Word Length: {features['scores']['word_length']}
  - Call to Action: {features['scores']['cta']}
  - Emoji Usage: {features['scores']['emoji']}
  - Compliance: {features['scores']['compliance']}

Please evaluate this copy considering the brand guidelines and scores provided."""

    def _get_assistant_message(self, features: Dict) -> str:
        return f"""Brand-Focused Analysis for {features['brand_name']}:

Overall Score: {features['scores']['overall']:.1f}/10

Brand Voice Assessment:
- Brand Alignment: {features['scores']['brand_alignment']}/10
- Tone Consistency: {features['scores']['tone']}/10
- Personalization: {features['scores']['personalization']}/10

Message Structure:
- Words: {features['metrics']['word_count']}
- Sentences: {features['metrics']['sentence_count']}
- Scroll Length: {features['metrics']['scroll_count']}

Performance Metrics:
1. Brand and Voice
- Brand Alignment: {features['scores']['brand_alignment']}/10
- Tone: {features['scores']['tone']}/10
- Personalization: {features['scores']['personalization']}/10

2. Engagement and Structure
- Engagement Appeal: {features['scores']['engagement']}/10
- Call to Action: {features['scores']['cta']}/10
- Emoji Usage: {features['scores']['emoji']}/10
- Readability: {features['scores']['readability_index']}/10

3. Compliance and Technical
- Compliance: {features['scores']['compliance']}/10
- Word Length: {features['scores']['word_length']}/10

Brand-Specific Guidelines and Recommendations:
{features['rules']['brand_alignment']}
{features['rules']['tone']}
{features['rules']['engagement']}"""

    def process_excel_to_jsonl(self, input_file: str, output_file: str = "training_data.jsonl") -> str:
        try:
            df = pd.read_excel(input_file)
            self.logger.info(f"Loaded {len(df)} rows from Excel")
            
            df = self.clean_dataframe(df)
            
            examples = []
            for idx, row in df.iterrows():
                try:
                    example = self._create_training_example(row)
                    examples.append(example)
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
                    
            self.logger.info(f"Created {len(examples)} training examples in {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error processing Excel file: {e}")
            raise

class SMSDataProcessor(MarketingDataProcessor):
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.dropna(subset=['Message Body', 'Sl No', 'Brand Name'])
        cleaned_df = cleaned_df.dropna(how='all')
        
        # SMS-specific cleaning
        if 'Link Attached to Message' in cleaned_df.columns:
            cleaned_df['Attachment (Yes/No)'] = cleaned_df['Link Attached to Message'].notna().map({True: 'Yes', False: 'No'})
            cleaned_df['Attachment Type'] = cleaned_df['Link Attached to Message'].notna().map({True: 'Link', False: ''})
        
        self.logger.info(f"Cleaned SMS dataframe from {len(df)} to {len(cleaned_df)} valid rows")
        return cleaned_df

    def _create_training_example(self, row: pd.Series) -> Dict:
        try:
            features = {
                'brand_name': row['Brand Name'],
                'message_body': row['Message Body'],
                'sent_date': str(row['Sent Date']),
                'receiver_name': str(row['Receiver Name']),
                'attachment': {
                    'has_attachment': 'Yes' if pd.notna(row.get('Link Attached to Message', '')) else 'No',
                    'type': 'Link' if pd.notna(row.get('Link Attached to Message', '')) else '',
                    'link': str(row.get('Link Attached to Message', ''))
                },
                'scores': {
                    'overall': round(float(row['Final Score']), 1) if pd.notna(row['Final Score']) else 0.0,
                    'readability_index': round(float(row['Readabilty Index Final Score'])) if pd.notna(row['Readabilty Index Final Score']) else 0,
                    'tone': round(float(row['Tone Final Score'])) if pd.notna(row['Tone Final Score']) else 0,
                    'word_length': round(float(row['Word Length Final Score'])) if pd.notna(row['Word Length Final Score']) else 0,
                    'personalization': round(float(row['Personalization Depth Final Score'])) if pd.notna(row['Personalization Depth Final Score']) else 0,
                    'cta': round(float(row['Call To Action Final Score'])) if pd.notna(row['Call To Action Final Score']) else 0,
                    'engagement': round(float(row['Engagement Appeal Final Score'])) if pd.notna(row['Engagement Appeal Final Score']) else 0,
                    'brand_alignment': round(float(row['Brand Alignment Final Score'])) if pd.notna(row['Brand Alignment Final Score']) else 0,
                    'compliance': round(float(row['Compliance Considerations LLM Score'])) if pd.notna(row['Compliance Considerations LLM Score']) else 0
                },
                'rules': {
                    'readability': str(row.get('Readability Index Rules', '')),
                    'tone': str(row.get('Tone Rules', '')),
                    'word_length': str(row.get('Word Length Rules', '')),
                    'personalization': str(row.get('Personalization Depth Rules', '')),
                    'cta': str(row.get('Call To Action Rules', '')),
                    'engagement': str(row.get('Engagement Appeal Rules', '')),
                    'brand_alignment': str(row.get('Brand Alignment Rules', '')),
                    'compliance': str(row.get('Compliance Rules', ''))
                },
                'metrics': {
                    'word_count': int(row['Word length']) if pd.notna(row['Word length']) else 0,
                    'sentence_count': int(row['Sentence Length']) if pd.notna(row['Sentence Length']) else 0
                }
            }
            
            return {
                "messages": [
                    {"role": "system", "content": self._get_system_message(features)},
                    {"role": "user", "content": self._get_human_message(features)},
                    {"role": "assistant", "content": self._get_assistant_message(features)}
                ]
            }
        except Exception as e:
            self.logger.warning(f"Error processing SMS row: {str(e)}")
            raise

    def _get_system_message(self, features: Dict) -> str:
        return f"""You are an expert marketing copywriter for {features['brand_name']}. Your role is to maintain and enhance the brand's unique voice and identity while creating SMS marketing promotions.

Evaluate and enhance SMS marketing copy while adhering to the brand guidelines and SMS character limitations."""

    def _get_human_message(self, features: Dict) -> str:
        return f"""Analyze this SMS marketing copy for {features['brand_name']}:

Content: {features['message_body']}

Message Details:
- Date: {features['sent_date']}
- Recipient: {features['receiver_name']}
- Link: {features['attachment']['link']}

Message Metrics:
- Word Count: {features['metrics']['word_count']}
- Sentences: {features['metrics']['sentence_count']}

Current Scores (0-10):
- Overall: {features['scores']['overall']:.1f}
- Brand Alignment: {features['scores']['brand_alignment']}
- Tone: {features['scores']['tone']}
- Personalization: {features['scores']['personalization']}
- Engagement Appeal: {features['scores']['engagement']}
- Other Metrics:
  - Readability: {features['scores']['readability_index']}
  - Word Length: {features['scores']['word_length']}
  - Call to Action: {features['scores']['cta']}
  - Compliance: {features['scores']['compliance']}

Please evaluate this SMS copy considering the brand guidelines, character limitations, and scores provided."""

    def _get_assistant_message(self, features: Dict) -> str:
        return f"""SMS Marketing Analysis for {features['brand_name']}:

Overall Score: {features['scores']['overall']:.1f}/10

Message Structure:
- Words: {features['metrics']['word_count']}
- Sentences: {features['metrics']['sentence_count']}
- Link Included: {features['attachment']['has_attachment']}

Performance Metrics:
1. Content Quality
- Readability: {features['scores']['readability_index']}/10
- Word Length: {features['scores']['word_length']}/10
- Call to Action: {features['scores']['cta']}/10

2. Engagement and Personalization
- Engagement Appeal: {features['scores']['engagement']}/10
- Personalization: {features['scores']['personalization']}/10
- Tone: {features['scores']['tone']}/10

3. Brand and Compliance
- Brand Alignment: {features['scores']['brand_alignment']}/10
- Compliance: {features['scores']['compliance']}/10

Key Guidelines and Recommendations:
{features['rules']['readability']}
{features['rules']['word_length']}
{features['rules']['engagement']}"""

class FineTuningExecutor:
    def __init__(self, api_key: str, config: Optional[FineTuningConfig] = None):
        openai.api_key = api_key
        self.client = openai
        self.config = config or FineTuningConfig()
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def start_finetuning(self, training_file: str) -> Dict:
        try:
            with open(training_file, 'rb') as f:
                file_upload = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )

            job = self.client.fine_tuning.jobs.create(
                training_file=file_upload.id,
                model=self.config.model_name
            )
            
            self.logger.info(f"Started fine-tuning job: {job.id}")
            
            start_time = time.time()
            while True:
                status = self.client.fine_tuning.jobs.retrieve(job.id)
                
                if status.status == "succeeded":
                    return {"status": "success", "model": status.fine_tuned_model}
                elif status.status in ["failed", "cancelled"]:
                    raise RuntimeError(f"Fine-tuning failed: {status.error}")
                    
                if time.time() - start_time > self.config.timeout:
                    raise TimeoutError("Fine-tuning exceeded timeout")
                    
                time.sleep(self.config.poll_interval)
                
        except Exception as e:
            self.logger.error(f"Fine-tuning error: {e}")
            raise

def get_processor(channel: Channel) -> MarketingDataProcessor:
    """Factory function to get the appropriate processor based on channel"""
    if channel == Channel.WHATSAPP:
        return MarketingDataProcessor(ScoreConfig())
    elif channel == Channel.SMS:
        return SMSDataProcessor(ScoreConfig())
    else:
        raise ValueError(f"Unsupported channel: {channel}")

def main():
    parser = argparse.ArgumentParser(description='Marketing Copy Fine-tuning Pipeline')
    parser.add_argument('input_file', help='Input Excel file path')
    parser.add_argument('--channel', required=True, choices=['whatsapp', 'sms'], 
                      help='Marketing channel (whatsapp or sms)')
    parser.add_argument('--fine-tune', action='store_true', help='Run fine-tuning after processing')
    parser.add_argument('--api-key', help='OpenAI API key for fine-tuning')
    args = parser.parse_args()

    try:
        channel = Channel.from_string(args.channel)
        processor = get_processor(channel)
        
        output_file = f"{channel.value}_training_data.jsonl"
        jsonl_file = processor.process_excel_to_jsonl(args.input_file, output_file)
        
        if args.fine_tune:
            if not args.api_key:
                raise ValueError("API key required for fine-tuning")
                
            config = FineTuningConfig()
            finetuner = FineTuningExecutor(api_key=args.api_key, config=config)
            result = finetuner.start_finetuning(jsonl_file)
            print(f"Fine-tuning completed: {result}")
        else:
            print(f"JSONL file created: {jsonl_file}")
            
    except Exception as e:
        print(f"Error in pipeline: {e}")
        logging.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
