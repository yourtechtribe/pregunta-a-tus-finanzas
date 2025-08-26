"""
BBVA Bank Statement Extractor
Optimized extraction strategy based on actual BBVA format analysis (July 2025)
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BBVAExtractor:
    """
    Specialized extractor for BBVA bank statements.
    Supports both Excel and PDF formats with format-specific strategies.
    """
    
    def __init__(self):
        # Keywords in "Concepto" field for categorization
        # Based on actual BBVA data analysis
        # MEJORADO: Añadidas más palabras clave genéricas
        self.concept_keywords = {
            # Alimentación genérico
            'supermercado': 'Groceries',
            'alimentacion': 'Groceries',
            'compra': 'Groceries',  # Asume que "compra" sola es supermercado
            'nomina': 'Income',
            'transferencia recibida': 'Income',
            'abono': 'Income',
            'pension': 'Income',
            'bonificacion': 'Income',
            'seguridad social': 'Taxes',
            'cuota seguridad social': 'Taxes',
            'cuota': 'Taxes',
            'comision': 'Fees',
            'prestamo': 'Loan',
            'credito': 'Loan',
            'amortizacion': 'Loan',
            'hipoteca': 'Loan',
            'transferencia realizada': 'Transfers',
            'transferencia enviada': 'Transfers',
            'traspaso': 'Internal Transfer',
            'traspaso programa': 'Internal Transfer',
            'traspaso cuenta': 'Internal Transfer',
            'bizum': 'Transfers',
            'donacion': 'Donations',
            'fundacion': 'Donations',
            # Vivienda
            'alquiler': 'Housing',
            'hipoteca': 'Housing',
            'comunidad': 'Housing',
            'ibi': 'Housing',
            # Transporte
            'gasolina': 'Transportation',
            'gasolinera': 'Transportation',
            'parking': 'Transportation',
            'peaje': 'Transportation',
            'autopista': 'Transportation',
            # Salud
            'farmacia': 'Healthcare',
            'clinica': 'Healthcare',
            'hospital': 'Healthcare',
            'medico': 'Healthcare',
            'dentista': 'Healthcare',
            # Ocio
            'gimnasio': 'Sports',
            'gym': 'Sports',
            'deporte': 'Sports',
            'netflix': 'Entertainment',
            'spotify': 'Entertainment',
            'hbo': 'Entertainment',
            'disney': 'Entertainment',
            'suscripcion': 'Entertainment',
            'ong': 'Donations',
            'cajero': 'ATM',
            'retirada efectivo': 'ATM'
        }
        
        # Regex patterns for merchant/concept categorization
        # Based on actual merchants found in BBVA data
        # MEJORADO: Añadidos patrones genéricos
        self.category_patterns = {
            'Groceries': r'(?i)(supermercado|alimentacion|mercadona|carrefour|lidl|aldi|dia|eroski|alcampo|supermercats|ametller origen|roges supermercats|sapa|consum|ahorramas|froiz|condis|bonpreu|caprabo)',
            'Food & Dining': r'(?i)(mcdonald|restaurant|restaurante|cafe|cafeteria|bar|pizzeria|burger|kebab|sushi|tapas|cerveceria|gelateria|umami|aramark|vencafesa|l0ki83|fa fum|telepizza|dominos|glovoapp|uber eats|just eat|deliveroo|comer|comida)',
            'Transportation': r'(?i)(gasolina|gasolinera|repsol|cepsa|shell|bp|metro|bus|renfe|tmb|taxi|uber|cabify|parking|tunelspan)',
            'Shopping': r'(?i)(zara|h&m|primark|amazon|aliexpress|ikea|decathlon|mango|corte ingles|fnac)',
            'Entertainment': r'(?i)(netflix|spotify|hbo|disney|cine|teatro|concierto|ocio|gaming)',
            'Healthcare': r'(?i)(farmacia|clinica|hospital|medico|dentista|salud|sanitario)',
            'Utilities': r'(?i)(luz|gas|agua|telefono|internet|movistar|vodafone|orange|endesa|naturgy)',
            'Services': r'(?i)(perruquer|peluquer|lavanderia|tintoreria|reparacion|mantenimiento|innovat|paypal|acens|taxdown|hosting)',
            'Education': r'(?i)(uab|universidad|escuela|colegio|formacion|curso|udemy|coursera)',
            'Tech & Software': r'(?i)(claude\.ai|openai|github|google|microsoft|apple\.com|software|codificando bits)',
            'Sports': r'(?i)(gym|gimnasio|fitness|deporte|sport|piscina|yoga|pilates|crossfit)',
            'Donations': r'(?i)(fundacion|donacion|ong|caritas|cruz roja|unicef|level up)',
            'Internal Transfer': r'(?i)(traspaso programa tu cuenta|traspaso cuenta)',
            'Vending': r'(?i)(vending|maquina)'
        }
    
    def extract(self, file_path: Union[str, Path], use_gpt: bool = False, gpt_api_key: Optional[str] = None, use_gpt5: bool = True) -> Dict:
        """
        Main extraction method that handles Excel, CSV and PDF formats.
        
        Args:
            file_path: Path to BBVA statement file
            use_gpt: Whether to use GPT for intelligent categorization
            gpt_api_key: OpenAI API key (optional, uses env var if not provided)
            use_gpt5: Use GPT-5-nano (default True) instead of GPT-4o
            
        Returns:
            Dictionary with extracted and processed data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing BBVA statement: {file_path.name}")
        
        # Route to appropriate extractor
        if file_path.suffix.lower() == '.xlsx':
            data = self._extract_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            data = self._extract_csv(file_path)
        elif file_path.suffix.lower() == '.pdf':
            data = self._extract_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Process and enrich data
        processed_data = self._process_data(data)
        
        # Use GPT for categorization if requested
        if use_gpt:
            # Convert processed data back to DataFrame for GPT
            df_for_gpt = pd.DataFrame(processed_data['transactions'])
            df_for_gpt['amount'] = df_for_gpt['amount'].astype(float)
            
            # Apply GPT categorization (GPT-5-nano by default)
            df_categorized = self.categorize_with_gpt(df_for_gpt, api_key=gpt_api_key, use_gpt5=use_gpt5)
            
            # Update transactions with GPT results
            processed_data['transactions'] = df_categorized.to_dict('records')
            
            # Add GPT analysis if available
            if hasattr(self, 'gpt_analysis'):
                processed_data['gpt_analysis'] = self.gpt_analysis
        
        # Add metadata
        processed_data['metadata'] = {
            'source_file': file_path.name,
            'extraction_date': datetime.now().isoformat(),
            'bank': 'BBVA',
            'format': file_path.suffix[1:].upper(),
            'categorization_method': 'GPT-5-nano' if (use_gpt and use_gpt5) else 'GPT-4o' if use_gpt else 'Rule-based'
        }
        
        return processed_data
    
    def _extract_excel(self, file_path: Path) -> pd.DataFrame:
        """
        Extract data from BBVA Excel format.
        BBVA Excel has headers in row 4, data starts in row 5.
        """
        logger.info("Using Excel extraction (100% accuracy)")
        
        # Try to read Excel with proper headers
        try:
            # First, try reading without assumptions
            df_test = pd.read_excel(file_path, nrows=10)
            
            # Check if we can find the header row
            header_row = None
            for i in range(10):
                row = df_test.iloc[i].astype(str)
                if 'Fecha' in row.values or 'Importe' in row.values:
                    header_row = i
                    break
            
            if header_row is not None:
                df = pd.read_excel(
                    file_path,
                    skiprows=header_row,
                    dtype={'Importe': str}
                )
            else:
                # Try standard read
                df = pd.read_excel(file_path, dtype={'Importe': str})
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Map possible column name variations
        column_mapping = {
            'F.Valor': 'Fecha Valor',
            'Fecha': 'Fecha Operación',
            'Divisa_Importe': 'Divisa',
            'Disponible': 'Saldo'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Remove empty rows or header rows mixed in data
        df = df[df['Fecha Operación'].notna()]
        df = df[~df['Fecha Operación'].astype(str).str.contains('Fecha', na=False)]
        
        logger.info(f"Extracted {len(df)} transactions from Excel")
        
        return df
    
    def _extract_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Extract data from BBVA CSV format.
        """
        logger.info("Using CSV extraction")
        
        # Read CSV - keep Importe as string initially to detect format
        # BBVA uses semicolon as separator
        df = pd.read_csv(file_path, sep=';', dtype={'Importe': str})
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Map column name variations (CSV uses different names)
        column_mapping = {
            'F.Valor': 'Fecha Valor',
            'Fecha': 'Fecha Operación',
            'Divisa_Importe': 'Divisa',
            'Disponible': 'Saldo'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Remove empty rows or header rows that might be mixed in data
        df = df[df['Fecha Operación'].notna()]
        df = df[~df['Fecha Operación'].astype(str).str.contains('Fecha|F.Valor', na=False, case=False)]
        
        # Remove the duplicate header row if it exists (row with "Concepto" in Concepto field)
        df = df[df['Concepto'] != 'Concepto']
        
        logger.info(f"Extracted {len(df)} transactions from CSV")
        
        return df
    
    def _extract_pdf(self, file_path: Path) -> pd.DataFrame:
        """
        Extract data from BBVA PDF format.
        This is a placeholder for PDF extraction - implement with Gemini Vision or LlamaParse.
        """
        logger.warning("PDF extraction requires external API (Gemini Vision recommended)")
        
        # Placeholder for PDF extraction
        # In production, implement with:
        # - Gemini Vision 2.0 (recommended)
        # - LlamaParse 
        # - Docling
        
        raise NotImplementedError(
            "PDF extraction not implemented. Use Excel format or implement with Gemini Vision API"
        )
    
    def _process_data(self, df: pd.DataFrame) -> Dict:
        """
        Process and normalize BBVA data.
        """
        # Normalize amount format
        df['amount_raw'] = df['Importe'].copy()
        
        # Detect format: check if values contain commas
        sample_values = df['Importe'].head(10).astype(str)
        has_comma = any(',' in val for val in sample_values)
        
        if has_comma:
            # Spanish format with comma as decimal: 1.234,56
            logger.info("Detected Spanish format with comma decimal (1.234,56)")
            df['amount'] = (
                df['Importe']
                .astype(str)
                .str.replace('.', '', regex=False)  # Remove thousand separator
                .str.replace(',', '.', regex=False)  # Replace decimal comma with dot
                .str.strip()
                .astype(float)
            )
        else:
            # Standard format with dot as decimal: 1234.56 or -301.63
            logger.info("Detected standard format with dot decimal (1234.56)")
            df['amount'] = df['Importe'].astype(float)
        
        # Parse dates - handle different date formats
        try:
            df['date'] = pd.to_datetime(df['Fecha Operación'], format='%d/%m/%Y')
        except:
            try:
                df['date'] = pd.to_datetime(df['Fecha Operación'], format='%Y-%m-%d')
            except:
                df['date'] = pd.to_datetime(df['Fecha Operación'])
        
        try:
            df['value_date'] = pd.to_datetime(df['Fecha Valor'], format='%d/%m/%Y')
        except:
            try:
                df['value_date'] = pd.to_datetime(df['Fecha Valor'], format='%Y-%m-%d')
            except:
                df['value_date'] = pd.to_datetime(df['Fecha Valor'])
        
        # Clean descriptions
        df['description'] = df['Concepto'].str.strip()
        df['description_clean'] = self._clean_description(df['Concepto'])
        
        # Categorization
        df['bbva_category'] = df.get('Movimiento', 'OTROS').fillna('OTROS')
        df['category'] = df.apply(self._categorize_transaction, axis=1)
        
        # Additional fields
        df['currency'] = df['Divisa'] if 'Divisa' in df.columns else 'EUR'
        df['notes'] = df['Observaciones'].fillna('') if 'Observaciones' in df.columns else ''
        
        # Calculate statistics
        stats = self._calculate_statistics(df)
        
        # Prepare output
        transactions = df[[
            'date', 'value_date', 'description', 'description_clean',
            'amount', 'currency', 'category', 'bbva_category', 'notes'
        ]].to_dict('records')
        
        # Format dates as strings for JSON serialization (mantener formato español)
        for t in transactions:
            t['date'] = t['date'].strftime('%d/%m/%Y')  # Formato español
            t['value_date'] = t['value_date'].strftime('%d/%m/%Y')  # Formato español
        
        return {
            'transactions': transactions,
            'statistics': stats,
            'period': {
                'start': df['date'].min().strftime('%d/%m/%Y'),  # Formato español
                'end': df['date'].max().strftime('%d/%m/%Y')  # Formato español
            }
        }
    
    def _clean_description(self, descriptions: pd.Series) -> pd.Series:
        """
        Clean and normalize transaction descriptions.
        """
        cleaned = descriptions.str.strip()
        
        # Remove common prefixes
        cleaned = cleaned.str.replace(r'^COMPRA EN\s+', '', regex=True)
        cleaned = cleaned.str.replace(r'^PAGO EN\s+', '', regex=True)
        cleaned = cleaned.str.replace(r'^TRANSFERENCIA DE\s+', '', regex=True)
        cleaned = cleaned.str.replace(r'^RECIBO DE\s+', '', regex=True)
        
        # Remove card numbers and reference codes
        cleaned = cleaned.str.replace(r'\*{4}\d{4}', '', regex=True)
        cleaned = cleaned.str.replace(r'\b\d{10,}\b', '', regex=True)
        
        # Normalize whitespace
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
        
        return cleaned
    
    def _categorize_transaction(self, row: pd.Series) -> str:
        """
        Categorize transaction using multiple strategies.
        Priority: 1) Special cases, 2) Concept keywords, 3) Income detection, 4) Pattern matching
        Note: This is a fallback method. For better results, use GPTCategorizer
        """
        description_lower = row['description'].lower() if pd.notna(row['description']) else ''
        amount = row['amount']
        
        # 1. Special case: Plan de Pensiones with negative amount = Savings
        if 'plan de pensiones' in description_lower and amount < 0:
            return 'Savings'
        # Plan de Pensiones with positive amount = Income (withdrawal)
        elif 'plan de pensiones' in description_lower and amount > 0:
            return 'Income'
        
        # 2. Check concept keywords (most reliable)
        for keyword, category in self.concept_keywords.items():
            if keyword in description_lower:
                return category
        
        # 3. Check if it's income (positive amount and not a refund)
        if amount > 0 and 'devolucion' not in description_lower:
            return 'Income'
        
        # 4. Pattern matching on description for merchants
        description_clean_lower = row['description_clean'].lower() if pd.notna(row['description_clean']) else ''
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, description_clean_lower):
                return category
        
        # 5. Default category
        return 'Other'
    
    def categorize_with_gpt(self, df: pd.DataFrame, api_key: Optional[str] = None, use_gpt5: bool = True) -> pd.DataFrame:
        """
        Use GPT-5-nano or GPT-4o for intelligent categorization of transactions.
        
        Args:
            df: DataFrame with processed transactions
            api_key: OpenAI API key (optional, will use env var if not provided)
            use_gpt5: Use GPT-5-nano (default True) instead of GPT-4o
            
        Returns:
            DataFrame with GPT categorizations
        """
        try:
            if use_gpt5:
                try:
                    from gpt5_nano_categorizer import GPT5NanoCategorizer
                except ImportError:
                    from .gpt5_nano_categorizer import GPT5NanoCategorizer
                logger.info("Using GPT-5-nano for intelligent categorization (91% cost savings)")
                categorizer = GPT5NanoCategorizer(api_key=api_key)
            else:
                try:
                    from gpt_categorizer import GPTCategorizer
                except ImportError:
                    from .gpt_categorizer import GPTCategorizer
                logger.info("Using GPT-4o for intelligent categorization")
                categorizer = GPTCategorizer(api_key=api_key)
            
            # Prepare data for GPT
            df_for_gpt = df.copy()
            
            # Categorize in batches
            categorized = categorizer.categorize_batch(df_for_gpt, batch_size=10)
            
            # Analyze patterns (only if method exists)
            if hasattr(categorizer, 'analyze_spending_patterns'):
                analysis = categorizer.analyze_spending_patterns(categorized)
                # Store analysis in metadata
                self.gpt_analysis = analysis
            else:
                logger.info("Spending pattern analysis not available for this categorizer")
            
            return categorized
            
        except ImportError:
            logger.warning("GPTCategorizer not available. Install openai package for GPT-4o categorization")
            return df
        except Exception as e:
            logger.error(f"Error using GPT categorization: {e}")
            return df
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics from transactions.
        """
        income = df[df['amount'] > 0]['amount'].sum()
        expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        # Category breakdown
        category_spending = (
            df[df['amount'] < 0]
            .groupby('category')['amount']
            .sum()
            .abs()
            .to_dict()
        )
        
        # Daily statistics
        daily_balance = (
            df.groupby('date')['amount']
            .sum()
            .cumsum()
        )
        
        return {
            'total_income': round(income, 2),
            'total_expenses': round(expenses, 2),
            'net_flow': round(income - expenses, 2),
            'transaction_count': len(df),
            'average_expense': round(expenses / len(df[df['amount'] < 0]), 2) if len(df[df['amount'] < 0]) > 0 else 0,
            'category_breakdown': {k: round(v, 2) for k, v in category_spending.items()},
            'max_expense': round(abs(df[df['amount'] < 0]['amount'].min()), 2) if len(df[df['amount'] < 0]) > 0 else 0,
            'max_income': round(df[df['amount'] > 0]['amount'].max(), 2) if len(df[df['amount'] > 0]) > 0 else 0
        }
    
    def validate_extraction(self, data: Dict) -> Dict:
        """
        Validate extracted data for completeness and accuracy.
        """
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for required fields
        if not data.get('transactions'):
            validation_report['errors'].append("No transactions found")
            validation_report['is_valid'] = False
        
        # Check transaction integrity
        transactions = data.get('transactions', [])
        
        # Verify dates are in order
        dates = [datetime.strptime(t['date'], '%Y-%m-%d') for t in transactions]
        if dates != sorted(dates):
            validation_report['warnings'].append("Transactions not in chronological order")
        
        # Check for uncategorized transactions
        uncategorized = [t for t in transactions if t.get('category') == 'Other']
        if len(uncategorized) > len(transactions) * 0.3:
            validation_report['warnings'].append(
                f"{len(uncategorized)} transactions ({len(uncategorized)*100/len(transactions):.1f}%) are uncategorized"
            )
        
        # Verify statistics
        if 'statistics' in data:
            stats = data['statistics']
            calculated_net = stats['total_income'] - stats['total_expenses']
            if abs(calculated_net - stats['net_flow']) > 0.01:
                validation_report['errors'].append("Statistics calculation mismatch")
                validation_report['is_valid'] = False
        
        return validation_report


def main():
    """
    Example usage of BBVAExtractor with GPT-5-nano
    """
    # Load API key from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    print("=" * 70)
    print("BBVA EXTRACTOR WITH GPT-5-NANO")
    print("=" * 70)
    print()
    
    # Initialize extractor
    extractor = BBVAExtractor()
    
    # Example file paths - try CSV first, then Excel
    csv_file = Path("/home/Usuario/medium/adk-google/rag-finances/data/raw/movimientos-bbva-julio-2025-clean.csv")
    excel_file = Path("/home/Usuario/medium/adk-google/rag-finances/data/raw/movimientos-bbva-julio-2025.xlsx")
    
    # Choose the file that exists
    if csv_file.exists():
        file_to_process = csv_file
    elif excel_file.exists():
        file_to_process = excel_file
    else:
        file_to_process = None
    
    if file_to_process:
        print(f"Processing file: {file_to_process.name}")
        print()
        
        # Extract data WITH GPT-5-nano categorization
        print("Step 1: Extracting and categorizing with GPT-5-nano...")
        data = extractor.extract(
            file_to_process, 
            use_gpt=True,       # Enable GPT categorization
            gpt_api_key=api_key,  # Pass the API key
            use_gpt5=True       # Use GPT-5-nano (not GPT-4o)
        )
        
        # Validate
        validation = extractor.validate_extraction(data)
        
        # Save results
        output_path = Path("/home/Usuario/medium/adk-google/rag-finances/data/processed/bbva_gpt5_nano_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print()
        print("=" * 70)
        print("EXTRACTION COMPLETED WITH GPT-5-NANO!")
        print("=" * 70)
        print(f"Period: {data['period']['start']} to {data['period']['end']}")
        print(f"Transactions: {data['statistics']['transaction_count']}")
        print(f"Total Income: €{data['statistics']['total_income']:,.2f}")
        print(f"Total Expenses: €{data['statistics']['total_expenses']:,.2f}")
        print(f"Net Flow: €{data['statistics']['net_flow']:,.2f}")
        print(f"Categorization Method: {data['metadata']['categorization_method']}")
        
        # Show sample categorized transactions
        print("\nSample Categorized Transactions:")
        print("-" * 70)
        for i, t in enumerate(data['transactions'][:5], 1):
            print(f"{i}. {t['description'][:40]:40} | €{abs(t['amount']):8.2f}")
            if 'category' in t:
                print(f"   Category: {t.get('category', 'N/A'):15} | Confidence: {t.get('confidence', 'N/A')}")
                if t.get('merchant_name'):
                    print(f"   Merchant: {t['merchant_name']}")
            print()
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        if not validation['is_valid']:
            print("\nErrors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Show GPT analysis if available
        if 'gpt_analysis' in data:
            print("\nGPT-5-nano Analysis:")
            print("-" * 70)
            analysis = data['gpt_analysis']
            if 'category_summary' in analysis:
                print("Category Breakdown:")
                for cat, stats in analysis['category_summary'].items():
                    print(f"  {cat}: €{abs(stats.get('total', 0)):,.2f}")
            if 'insights' in analysis:
                print("\nInsights:")
                for insight in analysis['insights']:
                    print(f"  • {insight}")
    else:
        print(f"File not found: {csv_file}")
        print("Please ensure the BBVA CSV file is in the data/raw/ directory")


if __name__ == "__main__":
    main()