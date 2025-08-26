"""
GPT-5-nano Transaction Categorizer (OFFICIAL)
Uses OpenAI's GPT-5-nano model released August 7, 2025
Official pricing: $0.05/1M input, $0.40/1M output
Context window: 400K tokens (272K input + 128K output)
"""

import os
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionCategory(str, Enum):
    """
    Financial transaction categories synchronized with BBVA extractor
    Matches exactly the categories used in bbva_extractor.py
    """
    # Alimentación y restauración
    GROCERIES = "Groceries"
    FOOD_DINING = "Food & Dining"
    
    # Transporte
    TRANSPORTATION = "Transportation"
    
    # Compras
    SHOPPING = "Shopping"
    
    # Entretenimiento y deportes
    ENTERTAINMENT = "Entertainment"
    SPORTS = "Sports"
    
    # Salud
    HEALTHCARE = "Healthcare"
    
    # Hogar y servicios
    HOUSING = "Housing"  # Alquiler, hipoteca, comunidad
    UTILITIES = "Utilities"  # Luz, gas, agua, internet
    SERVICES = "Services"  # Peluquería, lavandería, etc.
    
    # Educación y tecnología
    EDUCATION = "Education"
    TECH_SOFTWARE = "Tech & Software"
    
    # Finanzas
    INCOME = "Income"  # Nóminas, abonos
    SAVINGS = "Savings"  # Plan de pensiones, ahorro
    LOAN = "Loan"  # Préstamos, créditos
    TAXES = "Taxes"  # Impuestos, seguridad social
    FEES = "Fees"  # Comisiones bancarias
    
    # Transferencias
    TRANSFERS = "Transfers"  # Transferencias externas
    INTERNAL_TRANSFER = "Internal Transfer"  # Entre cuentas propias
    
    # Otros
    ATM = "ATM"  # Retiradas de efectivo
    DONATIONS = "Donations"  # Donaciones, caridad
    VENDING = "Vending"  # Máquinas expendedoras
    OTHER = "Other"  # Sin categorizar


class ConfidenceLevel(str, Enum):
    """Confidence level of the categorization"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TransactionClassification(BaseModel):
    """
    Structured output for a single transaction classification
    Optimized for GPT-5-nano minimal output
    """
    transaction_id: int = Field(description="Index or ID of the transaction")
    category: TransactionCategory = Field(description="Primary category")
    subcategory: Optional[str] = Field(default=None, description="Subcategory if applicable")
    merchant_name: Optional[str] = Field(default=None, description="Cleaned merchant name")
    merchant_type: Optional[str] = Field(default=None, description="Type of merchant")
    confidence: ConfidenceLevel = Field(description="Confidence level")
    reasoning: Optional[str] = Field(default=None, description="Brief explanation")
    tags: List[str] = Field(default_factory=list, description="Tags")
    is_recurring: bool = Field(default=False, description="Is recurring")
    is_essential: bool = Field(default=False, description="Is essential expense")


class BatchClassificationResponse(BaseModel):
    """
    Response containing multiple transaction classifications
    Optimized for GPT-5-nano batch processing
    """
    model_config = {"protected_namespaces": ()}
    
    classifications: List[TransactionClassification]
    processing_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "gpt-5-nano"
    total_processed: int
    

class GPT5NanoCategorizer:
    """
    Ultra-efficient transaction categorizer using GPT-5-nano
    Released: August 7, 2025
    91% cost savings vs GPT-4o for classification tasks
    Official API endpoint: Available in OpenAI API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the categorizer with OpenAI API
        
        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY env var
        """
        try:
            from openai import OpenAI
            import openai
        except ImportError:
            raise ImportError("Please install openai package: pip install openai>=1.0.0")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key)
        self.openai = openai
        
        # GPT-5-nano is officially available since August 7, 2025
        # Fallback chain: gpt-5-nano -> gpt-5-mini -> gpt-4o-mini
        self.model = self._get_best_model()
        
        # Optimized system prompt for GPT-5-nano with BBVA categories
        self.system_prompt = """You are a financial transaction categorizer specialized in Spanish BBVA bank statements.

TASK: Categorize each transaction using EXACT categories from the list.

CATEGORIES TO USE:
- Groceries (supermercados, alimentación)
- Food & Dining (restaurantes, cafeterías)
- Transportation (gasolina, parking, metro)
- Shopping (ropa, Amazon, compras online)
- Entertainment (Netflix, cine, ocio)
- Sports (gimnasio, deportes)
- Healthcare (farmacia, médico, clínica)
- Housing (alquiler, hipoteca, comunidad)
- Utilities (luz, gas, agua, internet)
- Services (peluquería, reparaciones)
- Education (universidad, cursos)
- Tech & Software (software, apps)
- Income (nómina, ingresos)
- Savings (plan pensiones, ahorro)
- Loan (préstamos, créditos)
- Taxes (impuestos, seguridad social)
- Fees (comisiones bancarias)
- Transfers (transferencias enviadas)
- Internal Transfer (entre cuentas propias)
- ATM (retirada efectivo)
- Donations (donaciones, ONG)
- Vending (máquinas expendedoras)
- Other (no categorizable)

RULES:
1. Use ONLY the categories listed above
2. Identify Spanish merchants and terms
3. Mark recurring transactions (subscriptions, monthly payments)
4. Flag essential expenses (groceries, utilities, housing, healthcare)

OUTPUT: Structured JSON only. Be concise."""
    
    def _get_best_model(self) -> str:
        """
        Detect and use the best available model
        Note: GPT-5 models may not be publicly available yet
        """
        models_priority = [
            "gpt-5-nano",         # Target model (if available)
            "gpt-5-mini",         # Alternative GPT-5 model
            "gpt-5",              # Full GPT-5 model
            "gpt-4o-mini",        # Reliable fallback: $0.15/1M input, $0.60/1M output
            "gpt-4o-2024-08-06",  # Alternative fallback
            "gpt-4o"              # Last resort
        ]
        
        try:
            # Try to list available models
            available_models = self.client.models.list()
            model_ids = {model.id for model in available_models.data}
            
            logger.info(f"Available models: {sorted(model_ids)}")
            
            for model in models_priority:
                if model in model_ids:
                    logger.info(f"Using model: {model}")
                    return model
                    
            logger.warning(f"None of the preferred models available. Available models: {sorted(model_ids)}")
            
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
        
        # Default fallback - use gpt-4o-mini which definitely supports structured outputs
        default_model = "gpt-4o-mini"
        logger.info(f"Using default model: {default_model}")
        return default_model
    
    def _is_reasoning_model(self) -> bool:
        """
        Check if the current model is a reasoning model (GPT-5 series, o1, o3, etc.)
        Reasoning models have different parameter support
        """
        reasoning_models = ["gpt-5", "o1", "o3"]
        return any(reasoning_model in self.model.lower() for reasoning_model in reasoning_models)
    
    def _get_model_parameters(self) -> dict:
        """
        Get appropriate parameters for the current model
        """
        base_params = {
            "model": self.model,
            "messages": [],  # Will be filled by caller
            "response_format": None,  # Will be filled by caller
        }
        
        if self._is_reasoning_model():
            # GPT-5/reasoning model parameters
            base_params.update({
                "max_completion_tokens": 8000,  # Increased to ensure output after reasoning
                # Reasoning-specific parameters to reduce reasoning token usage
                "reasoning_effort": "minimal",  # minimal, low, medium, high
                # "verbosity": "low",             # low, medium, high
            })
        else:
            # Traditional model parameters
            base_params.update({
                "max_tokens": 2000,
                "temperature": 0.1,
                "seed": 42,
            })
            
        return base_params
    
    def categorize_batch(self, transactions: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
        """
        Categorize a batch of transactions using GPT-5-nano
        Optimized for speed and cost with larger batch sizes
        
        Args:
            transactions: DataFrame with transaction data
            batch_size: Number of transactions per batch (20 recommended for GPT-5)
            
        Returns:
            DataFrame with added categorization columns
        """
        start_time = time.time()
        logger.info(f"Starting categorization of {len(transactions)} transactions with {self.model}")
        
        # Prepare transactions for processing
        results = []
        total_tokens_used = 0
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions.iloc[i:i+batch_size]
            batch_results, tokens = self._process_batch_optimized(batch, start_idx=i)
            results.extend(batch_results)
            total_tokens_used += tokens
            logger.info(f"Processed {min(i+batch_size, len(transactions))}/{len(transactions)} transactions")
        
        # Convert results to DataFrame
        classifications_df = pd.DataFrame([r.model_dump() for r in results])
        
        # Merge with original data
        categorized = transactions.copy()
        categorized = categorized.reset_index(drop=True)
        
        # Add categorization columns
        for col in ['category', 'subcategory', 'merchant_name', 'merchant_type', 
                    'confidence', 'reasoning', 'tags', 'is_recurring', 'is_essential']:
            if col in classifications_df.columns:
                categorized[col] = classifications_df[col]
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        estimated_cost = self._estimate_cost(total_tokens_used)
        
        logger.info(f"Completed in {processing_time:.2f}s")
        logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        logger.info(f"Cost per transaction: ${estimated_cost/len(transactions):.6f}")
        
        # Store metrics
        self.processing_metrics = {
            'processing_time': processing_time,
            'total_tokens': total_tokens_used,
            'estimated_cost': estimated_cost,
            'cost_per_transaction': estimated_cost/len(transactions),
            'transactions_per_second': len(transactions)/processing_time
        }
        
        return categorized
    
    def _process_batch_optimized(self, batch: pd.DataFrame, start_idx: int = 0) -> tuple:
        """
        Process a single batch with GPT-5-nano optimizations
        Returns results and token count
        """
        # Prepare compact transaction format for minimal tokens
        transactions_text = self._format_transactions_compact(batch, start_idx)
        
        # Optimized prompt for GPT-5-nano
        user_prompt = f"""Categorize these {len(batch)} transactions:

{transactions_text}

For each: category, subcategory, merchant, confidence, recurring?, essential?"""
        
        try:
            # Get appropriate parameters for the current model
            params = self._get_model_parameters()
            params["messages"] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            params["response_format"] = BatchClassificationResponse
            
            # Log model-specific configuration
            if self._is_reasoning_model():
                logger.info(f"Using reasoning model {self.model} with max_completion_tokens={params.get('max_completion_tokens')}")
            else:
                logger.info(f"Using traditional model {self.model} with temperature={params.get('temperature')}")
            
            # Make the API call
            completion = self.client.chat.completions.parse(**params)
            
            # Extract parsed response and token usage
            # For GPT-5 models, check for reasoning tokens in completion_tokens_details
            tokens_used = 0
            if hasattr(completion, 'usage') and completion.usage:
                tokens_used = completion.usage.total_tokens
                # Log reasoning tokens if available (GPT-5 specific)
                if hasattr(completion.usage, 'completion_tokens_details'):
                    details = completion.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens
                        logger.info(f"Reasoning tokens used: {reasoning_tokens}")
                        # Adjust token accounting if needed
                        if reasoning_tokens and reasoning_tokens > 0:
                            logger.info(f"Total: {tokens_used}, Reasoning: {reasoning_tokens}, Output: {tokens_used - reasoning_tokens}")
            
            # Handle structured output response
            if completion.choices[0].message.parsed:
                parsed_response = completion.choices[0].message.parsed
                if hasattr(parsed_response, 'classifications'):
                    return parsed_response.classifications, tokens_used
                else:
                    logger.error(f"Parsed response missing 'classifications' field: {parsed_response}")
                    return self._create_fallback_classifications(batch, start_idx), tokens_used
            else:
                # Check if there's a refusal or other issue
                choice = completion.choices[0]
                if hasattr(choice.message, 'refusal') and choice.message.refusal:
                    logger.error(f"Model refused request: {choice.message.refusal}")
                else:
                    logger.error(f"Failed to parse response. Raw content: {choice.message.content}")
                return self._create_fallback_classifications(batch, start_idx), tokens_used
                
        except Exception as e:
            logger.error(f"Error calling {self.model}: {e}")
            # Log more specific error information for debugging
            if "max_tokens" in str(e).lower():
                logger.error("Token limit error - try reducing batch_size or increasing max_completion_tokens")
            elif "temperature" in str(e).lower():
                logger.error("Temperature parameter not supported for reasoning models")
            elif "invalid_request_error" in str(e).lower():
                logger.error(f"API request error: {e}")
            return self._create_fallback_classifications(batch, start_idx), 0
    
    def _format_transactions_compact(self, batch: pd.DataFrame, start_idx: int) -> str:
        """
        Format transactions in ultra-compact format for minimal token usage
        """
        lines = []
        for idx, row in batch.iterrows():
            transaction_id = start_idx + idx
            date = str(row.get('date', 'N/A'))[:10]  # Just date, no time
            desc = row.get('description', row.get('Concepto', ''))[:50]  # Limit length
            amt = row.get('amount', row.get('Importe', 0))
            
            # Ultra-compact format
            sign = "+" if amt > 0 else "-"
            lines.append(f"{transaction_id}|{date}|{desc}|{sign}€{abs(amt):.2f}")
        
        return "\n".join(lines)
    
    def _estimate_cost(self, tokens: int) -> float:
        """
        Estimate cost based on OFFICIAL model pricing
        """
        # OFFICIAL pricing per 1M tokens (August 7, 2025 release)
        pricing = {
            "gpt-5-nano": {"input": 0.05, "output": 0.40},     # OFFICIAL
            "gpt-5-mini": {"input": 0.25, "output": 2.00},     # OFFICIAL
            "gpt-5": {"input": 1.25, "output": 10.00},         # OFFICIAL
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00}
        }
        
        model_pricing = pricing.get(self.model, pricing["gpt-4o-mini"])
        
        # Rough estimate: 70% input, 30% output
        input_tokens = tokens * 0.7
        output_tokens = tokens * 0.3
        
        cost = (input_tokens * model_pricing["input"] / 1_000_000 + 
                output_tokens * model_pricing["output"] / 1_000_000)
        
        return cost
    
    def _create_fallback_classifications(self, batch: pd.DataFrame, start_idx: int) -> List[TransactionClassification]:
        """
        Create fallback classifications when API fails
        """
        classifications = []
        
        for idx, row in batch.iterrows():
            amount = row.get('amount', row.get('Importe', 0))
            description = str(row.get('description', row.get('Concepto', ''))).lower()
            
            # Simple rule-based fallback
            if amount > 0:
                category = TransactionCategory.INCOME
            elif any(word in description for word in ['mercadona', 'carrefour', 'supermercado']):
                category = TransactionCategory.GROCERIES
            elif any(word in description for word in ['restaurante', 'mcdonald', 'cafe', 'bar']):
                category = TransactionCategory.FOOD_DINING
            elif 'transferencia' in description:
                category = TransactionCategory.TRANSFERS
            else:
                category = TransactionCategory.OTHER
            
            classifications.append(TransactionClassification(
                transaction_id=start_idx + idx,
                category=category,
                confidence=ConfidenceLevel.LOW,
                reasoning="Fallback categorization",
                tags=[],
                is_recurring=False,
                is_essential=category in [TransactionCategory.GROCERIES, TransactionCategory.UTILITIES]
            ))
        
        return classifications
    
    def compare_with_gpt4o(self, gpt4o_results_path: str) -> Dict[str, Any]:
        """
        Compare GPT-5-nano results with GPT-4o results
        """
        # Load GPT-4o results
        with open(gpt4o_results_path, 'r', encoding='utf-8') as f:
            gpt4o_data = json.load(f)
        
        comparison = {
            'model_used': self.model,
            'gpt5_metrics': self.processing_metrics if hasattr(self, 'processing_metrics') else {},
            'gpt4o_metadata': gpt4o_data.get('metadata', {}),
            'cost_reduction': None,
            'speed_improvement': None,
            'analysis_comparison': {}
        }
        
        # Calculate cost reduction
        if 'gpt5_metrics' in comparison and comparison['gpt5_metrics']:
            gpt5_cost = comparison['gpt5_metrics']['estimated_cost']
            # Estimate GPT-4o cost (assuming similar token usage)
            gpt4o_cost = comparison['gpt5_metrics']['total_tokens'] * 2.50 / 1_000_000  # GPT-4o pricing
            comparison['cost_reduction'] = f"{((gpt4o_cost - gpt5_cost) / gpt4o_cost * 100):.1f}%"
        
        # Compare analysis results if available
        if 'analysis' in gpt4o_data:
            comparison['gpt4o_analysis'] = gpt4o_data['analysis']
        
        return comparison


def main():
    """
    Example usage with BBVA data - comparing with GPT-4o results
    """
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load sample data
    csv_file = Path("/home/Usuario/medium/adk-google/rag-finances/data/raw/movimientos-bbva-julio-2025-clean.csv")
    gpt4o_results = Path("/home/Usuario/medium/adk-google/rag-finances/data/processed/categorized_gpt4o.json")
    
    if not csv_file.exists():
        print(f"Sample file not found: {csv_file}")
        return
    
    # Read transactions
    df = pd.read_csv(csv_file)
    
    # Clean and prepare data
    df = df[df['Concepto'] != 'Concepto']  # Remove header rows
    
    # Convert Importe to string first, then replace comma and convert to float
    df['Importe'] = df['Importe'].astype(str)
    df['amount'] = df['Importe'].str.replace(',', '.').astype(float)
    df['description'] = df['Concepto']
    df['date'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d', errors='coerce')
    
    # Select first 10 transactions for demo (same as GPT-4o test)
    sample = df.head(10)
    
    print("=" * 60)
    print("Categorizing transactions with GPT-5-nano...")
    print("=" * 60)
    
    # Initialize categorizer
    categorizer = GPT5NanoCategorizer()
    
    # Start timing
    start_time = time.time()
    
    # Categorize transactions
    categorized = categorizer.categorize_batch(sample, batch_size=10)
    
    # Calculate timing
    processing_time = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    
    for idx, row in categorized.iterrows():
        print(f"\nTransaction {idx + 1}:")
        print(f"  Description: {row['description']}")
        print(f"  Amount: €{abs(row['amount']):.2f} {'(income)' if row['amount'] > 0 else '(expense)'}")
        print(f"  Category: {row.get('category', 'N/A')}")
        print(f"  Subcategory: {row.get('subcategory', 'N/A')}")
        print(f"  Merchant: {row.get('merchant_name', 'N/A')}")
        print(f"  Confidence: {row.get('confidence', 'N/A')}")
        print(f"  Essential: {row.get('is_essential', 'N/A')}")
        print(f"  Recurring: {row.get('is_recurring', 'N/A')}")
    
    # Performance metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS:")
    print("=" * 60)
    
    if hasattr(categorizer, 'processing_metrics'):
        metrics = categorizer.processing_metrics
        print(f"Model Used: {categorizer.model}")
        print(f"Processing Time: {metrics['processing_time']:.2f} seconds")
        print(f"Transactions/Second: {metrics['transactions_per_second']:.2f}")
        print(f"Total Tokens Used: {metrics['total_tokens']}")
        print(f"Estimated Cost: ${metrics['estimated_cost']:.4f}")
        print(f"Cost per Transaction: ${metrics['cost_per_transaction']:.6f}")
    
    # Compare with GPT-4o if results exist
    if gpt4o_results.exists():
        print("\n" + "=" * 60)
        print("COMPARISON WITH GPT-4o:")
        print("=" * 60)
        
        comparison = categorizer.compare_with_gpt4o(str(gpt4o_results))
        
        if comparison['cost_reduction']:
            print(f"Cost Reduction: {comparison['cost_reduction']}")
        
        print(f"Speed: {processing_time:.2f}s (GPT-5) vs ~3-5s (GPT-4o typical)")
        
        # Load GPT-4o results for detailed comparison
        with open(gpt4o_results, 'r', encoding='utf-8') as f:
            gpt4o_data = json.load(f)
        
        # Compare categorizations
        print("\nCategory Comparison (First 3 transactions):")
        for i in range(min(3, len(categorized))):
            gpt5_cat = categorized.iloc[i].get('category', 'N/A')
            gpt4o_cat = gpt4o_data['transactions'][i].get('category', 'N/A')
            match = "✓" if gpt5_cat == gpt4o_cat else "✗"
            print(f"  Transaction {i+1}: GPT-5: {gpt5_cat} | GPT-4o: {gpt4o_cat} [{match}]")
    
    # Save results
    output_file = Path("/home/Usuario/medium/adk-google/rag-finances/data/processed/categorized_gpt5_nano.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    categorized_dict = categorized.to_dict('records')
    
    # Prepare output data
    output_data = {
        'transactions': categorized_dict,
        'metadata': {
            'processed_date': datetime.now().isoformat(),
            'model': categorizer.model,
            'total_transactions': len(categorized),
            'processing_time_seconds': processing_time
        }
    }
    
    # Add metrics if available
    if hasattr(categorizer, 'processing_metrics'):
        output_data['performance_metrics'] = categorizer.processing_metrics
    
    # Add comparison if available
    if gpt4o_results.exists():
        output_data['comparison'] = comparison
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    
    if "gpt-5" in categorizer.model:
        print(f"✓ {categorizer.model} is NOW AVAILABLE for production!")
        print("  Released: August 7, 2025")
        print("  - 91% cost reduction vs GPT-4o")
        print("  - 400K token context window (3x larger)")
        print("  - 90% cache discount on repeated prompts")
        print("  - Ideal for classification and extraction tasks")
    else:
        print(f"⚠ Using {categorizer.model} (Check API access for GPT-5)")
        print("  GPT-5-nano is officially available since August 7, 2025")
        print("  Benefits of upgrading:")
        print("  - $0.05/1M input tokens (vs $2.50 for GPT-4o)")
        print("  - Better accuracy for financial categorization")


if __name__ == "__main__":
    main()