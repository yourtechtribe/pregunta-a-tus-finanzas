#!/usr/bin/env python3
"""
Adaptive Chunking Strategy for Financial RAG Pipeline

Implements multiple chunking strategies:
1. Narrative chunks - Rich contextual descriptions for better entity extraction
2. Time-based chunks - Daily/weekly/monthly aggregations
3. Category-based chunks - Grouped by spending categories
4. Merchant-based chunks - Grouped by entities/merchants
5. Pattern-based chunks - Recurring transactions and anomalies

Integrated with the anonymization and categorization pipeline.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from collections import defaultdict
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdaptiveChunkGenerator:
    """Generate multiple types of chunks from financial transactions"""
    
    CHUNK_STRATEGIES = [
        'narrative',      # Rich text descriptions
        'temporal',       # Time-based aggregations
        'categorical',    # Category groupings
        'entity',         # Merchant/entity based
        'pattern'         # Recurring and anomaly detection
    ]
    
    def __init__(self, 
                 json_path: str = None, 
                 csv_path: str = None,
                 anonymized: bool = True,
                 strategies: List[str] = None):
        """
        Initialize chunk generator
        
        Args:
            json_path: Path to processed transactions JSON
            csv_path: Path to raw CSV (legacy support)
            anonymized: Whether data has been anonymized
            strategies: List of strategies to use (default: all)
        """
        self.json_path = json_path
        self.csv_path = csv_path
        self.anonymized = anonymized
        self.strategies = strategies or self.CHUNK_STRATEGIES
        
        self.df = None
        self.transactions_data = None
        self.chunks = []
        self.chunk_stats = defaultdict(int)
        
        # Analysis structures
        self.merchant_profiles = defaultdict(lambda: {
            'total_spent': 0,
            'transaction_count': 0,
            'categories': set(),
            'typical_amount': [],
            'dates': [],
            'is_recurring': False
        })
        
        self.category_profiles = defaultdict(lambda: {
            'total_spent': 0,
            'transaction_count': 0,
            'merchants': set(),
            'daily_avg': 0,
            'weekly_pattern': defaultdict(float)
        })
        
        self.temporal_patterns = {
            'weekday_spending': defaultdict(float),
            'monthly_trends': defaultdict(float),
            'peak_spending_hours': defaultdict(int)
        }
        
    def load_data(self):
        """Load and prepare transaction data from JSON or CSV"""
        if self.json_path and Path(self.json_path).exists():
            logger.info(f"Loading data from JSON: {self.json_path}")
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, dict) and 'transactions' in data:
                transactions = data['transactions']
            elif isinstance(data, list):
                transactions = data
            else:
                transactions = data.get('data', [])
            
            self.df = pd.DataFrame(transactions)
            
            # Normalize column names
            self._normalize_columns()
            
            logger.info(f"Loaded {len(self.df)} transactions from JSON")
        elif self.csv_path and Path(self.csv_path).exists():
            logger.info(f"Loading data from CSV: {self.csv_path}")
            # Try different separators
            for sep in [';', ',', '\t']:
                try:
                    self.df = pd.read_csv(self.csv_path, sep=sep, encoding='utf-8')
                    if len(self.df.columns) > 1:
                        break
                except:
                    continue
            
            self._normalize_columns()
            logger.info(f"Loaded {len(self.df)} transactions from CSV")
        else:
            raise ValueError("No valid data source provided")
        
        # Clean and prepare data
        self._clean_data()
        
        # Build profiles for analysis
        self._build_profiles()
        
        logger.info(f"Data loaded: {len(self.df)} transactions ready for chunking")
        
    def _normalize_columns(self):
        """Normalize column names across different formats"""
        # Map common column variations
        column_mapping = {
            'date': 'Fecha',
            'Date': 'Fecha',
            'fecha': 'Fecha',
            'amount': 'Importe',
            'Amount': 'Importe',
            'importe': 'Importe',
            'description': 'Concepto',
            'Description': 'Concepto',
            'concepto': 'Concepto',
            'balance': 'Disponible',
            'Balance': 'Disponible',
            'disponible': 'Disponible',
            'category': 'Categoria',
            'Category': 'Categoria'
        }
        
        # Rename columns
        self.df.rename(columns=column_mapping, inplace=True)
        
        # Ensure required columns exist
        required = ['Fecha', 'Importe', 'Concepto']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _clean_data(self):
        """Clean and prepare transaction data"""
        # Convert date columns
        for date_col in ['Fecha', 'F.Valor']:
            if date_col in self.df.columns:
                self.df[date_col] = pd.to_datetime(
                    self.df[date_col], 
                    format='%d/%m/%Y',
                    errors='coerce'
                )
                # Try other formats if needed
                mask = self.df[date_col].isna()
                if mask.any():
                    self.df.loc[mask, date_col] = pd.to_datetime(
                        self.df.loc[mask, date_col],
                        format='%Y-%m-%d',
                        errors='coerce'
                    )
        
        # Convert numeric columns
        self.df['Importe'] = pd.to_numeric(self.df['Importe'], errors='coerce')
        if 'Disponible' in self.df.columns:
            self.df['Disponible'] = pd.to_numeric(self.df['Disponible'], errors='coerce')
        
        # Remove invalid rows
        self.df = self.df.dropna(subset=['Fecha', 'Importe'])
        
        # Sort by date
        self.df = self.df.sort_values('Fecha')
        
        # Add derived columns
        self.df['Weekday'] = self.df['Fecha'].dt.dayofweek
        self.df['Month'] = self.df['Fecha'].dt.month
        self.df['Year'] = self.df['Fecha'].dt.year
        self.df['Week'] = self.df['Fecha'].dt.isocalendar().week
        self.df['IsWeekend'] = self.df['Weekday'] >= 5
        self.df['IsIncome'] = self.df['Importe'] > 0
    
    def _build_profiles(self):
        """Build comprehensive profiles for analysis"""
        logger.info("Building transaction profiles...")
        
        for idx, row in self.df.iterrows():
            # Extract merchant
            merchant = self._extract_merchant(row['Concepto'])
            category = row.get('Categoria', self._categorize(row))
            amount = abs(row['Importe'])
            
            # Update merchant profile
            if merchant and not row['IsIncome']:
                profile = self.merchant_profiles[merchant]
                profile['total_spent'] += amount
                profile['transaction_count'] += 1
                profile['typical_amount'].append(amount)
                profile['categories'].add(category)
                profile['dates'].append(row['Fecha'])
            
            # Update category profile
            if not row['IsIncome']:
                cat_profile = self.category_profiles[category]
                cat_profile['total_spent'] += amount
                cat_profile['transaction_count'] += 1
                if merchant:
                    cat_profile['merchants'].add(merchant)
                cat_profile['weekly_pattern'][row['Weekday']] += amount
            
            # Update temporal patterns
            if not row['IsIncome']:
                self.temporal_patterns['weekday_spending'][row['Weekday']] += amount
                month_key = f"{row['Year']}-{row['Month']:02d}"
                self.temporal_patterns['monthly_trends'][month_key] += amount
        
        # Detect recurring transactions
        self._detect_recurring_transactions()
        
        logger.info(f"Built profiles: {len(self.merchant_profiles)} merchants, "
                   f"{len(self.category_profiles)} categories")
    
    def _detect_recurring_transactions(self):
        """Identify recurring transactions based on patterns"""
        for merchant, profile in self.merchant_profiles.items():
            if len(profile['dates']) < 2:
                continue
            
            # Sort dates and calculate intervals
            dates = sorted(profile['dates'])
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            if not intervals:
                continue
            
            # Check for monthly pattern (28-32 days)
            monthly = sum(1 for i in intervals if 28 <= i <= 32)
            if monthly >= len(intervals) * 0.6:  # 60% threshold
                profile['is_recurring'] = True
                profile['recurrence_type'] = 'monthly'
            
            # Check for weekly pattern (6-8 days)
            weekly = sum(1 for i in intervals if 6 <= i <= 8)
            if weekly >= len(intervals) * 0.6:
                profile['is_recurring'] = True
                profile['recurrence_type'] = 'weekly'
    
    def _extract_merchant(self, description: str) -> str:
        """Extract merchant/entity name from description
        
        TODO: Future Enhancement - LLM Integration
        ----------------------------------------
        Integrate with lightweight LLM (e.g., GPT-4o-mini, gpt-5-nano) for:
        1. Better merchant name extraction from complex descriptions
        2. Merchant name normalization (e.g., "AMZN" -> "Amazon")
        3. Category inference from merchant names
        4. Multi-language support for international transactions
        
        Example implementation:
        ```python
        if self.use_llm and self.llm_client:
            prompt = f"Extract merchant name from: {description}"
            merchant = self.llm_client.extract_merchant(prompt)
            if merchant:
                return merchant
        ```
        """
        # Common patterns in bank statements
        desc_lower = description.lower()
        
        # Known merchants/entities - Spanish market focus
        merchants = {
            'mcdonald': 'McDonald\'s',
            'carrefour': 'Carrefour',
            'zara': 'Zara',
            'amazon': 'Amazon',
            'netflix': 'Netflix',
            'spotify': 'Spotify',
            'mercadona': 'Mercadona',
            'dia': 'Dia',
            'lidl': 'Lidl',
            'ikea': 'IKEA',
            'decathlon': 'Decathlon',
            'el corte ingles': 'El Corte Inglés',
            'mediamarkt': 'MediaMarkt',
            'fnac': 'FNAC',
            'h&m': 'H&M',
            'primark': 'Primark',
            # Add more Spanish merchants
            'bbva': 'BBVA',
            'santander': 'Santander',
            'endesa': 'Endesa',
            'naturgy': 'Naturgy',
            'vodafone': 'Vodafone',
            'movistar': 'Movistar'
        }
        
        for pattern, name in merchants.items():
            if pattern in desc_lower:
                return name
                
        # Extract from common Spanish bank patterns
        if 'pago con tarjeta' in desc_lower or 'compra tarjeta' in desc_lower:
            # Try to extract merchant from card payment
            parts = description.split()
            if len(parts) > 2:  # Skip generic words
                for word in parts[2:]:  # Start after "PAGO CON"
                    if len(word) > 3 and word.upper() != 'TARJETA':
                        return word.title()
        
        # Handle transfers
        if 'transferencia' in desc_lower:
            return 'Transfer'
        
        # Return first meaningful word as merchant
        words = description.split()
        for word in words:
            if len(word) > 3 and not word.lower() in ['pago', 'con', 'tarjeta', 'compra']:
                return word.title()
        
        return 'Unknown'
    
    def _categorize(self, row) -> str:
        """Get category from preprocessed data
        
        This function relies on categories already assigned by the pipeline's
        categorization phase (RuleCategorizer + MultiAgentCategorizer).
        
        The chunking phase should NOT re-categorize transactions, as this would:
        1. Duplicate work already done
        2. Potentially create inconsistencies
        3. Ignore the sophisticated multi-agent categorization
        
        Categories come from:
        - 'category' field (from pipeline JSON)
        - 'Categoria' field (normalized column name)
        """
        # Priority 1: Check for category from pipeline
        if 'category' in row and pd.notna(row.get('category')):
            return row['category']
        
        # Priority 2: Check for normalized column name
        if 'Categoria' in row and pd.notna(row.get('Categoria')):
            return row['Categoria']
        
        # Priority 3: Check for category_name (some formats use this)
        if 'category_name' in row and pd.notna(row.get('category_name')):
            return row['category_name']
        
        # Fallback: Mark as uncategorized for logging
        # This should rarely happen if pipeline is working correctly
        logger.warning(f"Transaction without category found: {row.get('Concepto', row.get('description', 'Unknown'))}")
        
        # Return a generic category based on amount sign
        amount = row.get('Importe', row.get('amount', 0))
        if amount > 0:
            return 'Uncategorized Income'
        else:
            return 'Uncategorized Expense'
    
    def _generate_transaction_narrative(self, row, context: Dict) -> str:
        """Generate narrative text for a single transaction"""
        date = row['Fecha'].strftime('%B %d, %Y')
        weekday = row['Fecha'].strftime('%A')
        amount = row['Importe']
        description = row['Concepto']
        # Get category from row data if available, otherwise categorize
        if 'category' in row:
            category = row['category']
        else:
            category = self._categorize(row)
        merchant = self._extract_merchant(description)
        
        # Build narrative based on transaction type
        narrative_parts = []
        
        # Opening context
        narrative_parts.append(f"On {weekday}, {date}, a financial transaction occurred.")
        
        # Transaction details
        if amount > 0:
            # Income narrative
            narrative_parts.append(f"This was an income transaction of €{amount:.2f}.")
            
            if 'pension' in description.lower():
                narrative_parts.append(f"The income came from a pension payment, which is a regular monthly income source.")
            elif merchant:
                narrative_parts.append(f"The funds were received from {merchant}.")
            else:
                narrative_parts.append(f"The transaction was described as '{description}'.")
                
            narrative_parts.append(f"This income has been categorized as {category}.")
            
        else:
            # Expense narrative
            expense_amount = abs(amount)
            narrative_parts.append(f"This was an expense of €{expense_amount:.2f}.")
            
            if merchant:
                narrative_parts.append(f"The payment was made to {merchant}, ")
                
                # Add merchant context if available
                if merchant in self.merchant_profiles:
                    profile = self.merchant_profiles[merchant]
                    avg_amount = sum(profile['typical_amount']) / len(profile['typical_amount']) if profile['typical_amount'] else expense_amount
                    
                    if expense_amount > avg_amount * 1.5:
                        narrative_parts.append(f"which is higher than the typical amount of €{avg_amount:.2f} spent there. ")
                    elif expense_amount < avg_amount * 0.5:
                        narrative_parts.append(f"which is lower than the typical amount of €{avg_amount:.2f} spent there. ")
                    else:
                        narrative_parts.append(f"which is a typical purchase amount for this merchant. ")
                    
                    if profile['transaction_count'] > 1:
                        narrative_parts.append(f"This is transaction number {context.get('merchant_visit_count', 1)} with {merchant} this month. ")
            else:
                narrative_parts.append(f"The transaction was for '{description}'. ")
            
            narrative_parts.append(f"This expense falls under the {category} category.")
            
            # Add spending pattern context
            if 'monthly_category_total' in context:
                month_total = context['monthly_category_total'].get(category, 0)
                if month_total > 0:
                    narrative_parts.append(f" So far this month, €{month_total:.2f} has been spent on {category}.")
        
        # Add balance context
        if 'Disponible' in row and pd.notna(row['Disponible']):
            balance = row['Disponible']
            narrative_parts.append(f" After this transaction, the account balance was €{balance:.2f}.")
            
            # Add balance health indicator
            if balance < 500:
                narrative_parts.append(" The balance is running low and requires attention.")
            elif balance > 5000:
                narrative_parts.append(" The account maintains a healthy balance.")
        
        # Add temporal context
        if context.get('is_weekend', False):
            narrative_parts.append(" This was a weekend transaction.")
        
        if context.get('is_month_end', False):
            narrative_parts.append(" This transaction occurred near the end of the month.")
        
        # Add any additional notes
        if 'Observaciones' in row and pd.notna(row['Observaciones']) and row['Observaciones']:
            obs = row['Observaciones']
            # Anonymize sensitive data
            obs = obs.replace(row['Observaciones'][:20], "****") if len(obs) > 20 else "****"
            narrative_parts.append(f" Additional details: {obs}")
        
        return " ".join(narrative_parts)
    
    def _generate_daily_summary_narrative(self, date: datetime, transactions: List) -> str:
        """Generate narrative summary for a day's transactions"""
        weekday = date.strftime('%A')
        date_str = date.strftime('%B %d, %Y')
        
        total_income = sum(t['Importe'] for t in transactions if t['Importe'] > 0)
        total_expenses = abs(sum(t['Importe'] for t in transactions if t['Importe'] < 0))
        net_flow = total_income - total_expenses
        
        narrative = [f"Financial Summary for {weekday}, {date_str}:"]
        narrative.append(f"On this day, there were {len(transactions)} financial transactions.")
        
        if total_income > 0:
            narrative.append(f"Total income received was €{total_income:.2f}.")
        
        if total_expenses > 0:
            narrative.append(f"Total expenses amounted to €{total_expenses:.2f}.")
            
            # Categorize expenses
            expense_categories = defaultdict(float)
            for t in transactions:
                if t['Importe'] < 0:
                    category = self._categorize(t)
                    expense_categories[category] += abs(t['Importe'])
            
            if expense_categories:
                narrative.append("Expenses were distributed across the following categories:")
                for cat, amount in sorted(expense_categories.items(), key=lambda x: x[1], reverse=True):
                    narrative.append(f"  - {cat}: €{amount:.2f}")
        
        # Net flow analysis
        if net_flow > 0:
            narrative.append(f"The day ended with a positive cash flow of €{net_flow:.2f}.")
        elif net_flow < 0:
            narrative.append(f"The day ended with a negative cash flow of €{abs(net_flow):.2f}.")
        else:
            narrative.append("Income and expenses were perfectly balanced.")
        
        # Add context about spending patterns
        if total_expenses > 500:
            narrative.append("This was a high-spending day.")
        elif total_expenses < 50 and total_expenses > 0:
            narrative.append("Spending was minimal on this day.")
        
        return " ".join(narrative)
    
    def _generate_weekly_summary_narrative(self, week_data: Dict) -> str:
        """Generate narrative summary for a week"""
        narrative = [f"Weekly Financial Summary for Week {week_data['week_num']} of {week_data['year']}:"]
        
        narrative.append(f"During this week, from {week_data['start_date']} to {week_data['end_date']}, ")
        narrative.append(f"there were {week_data['transaction_count']} transactions.")
        
        if week_data['total_income'] > 0:
            narrative.append(f"Total weekly income was €{week_data['total_income']:.2f}.")
        
        if week_data['total_expenses'] > 0:
            narrative.append(f"Total weekly expenses were €{week_data['total_expenses']:.2f}.")
        
        # Top merchants
        if week_data['top_merchants']:
            narrative.append("The most frequented merchants this week were:")
            for merchant, data in week_data['top_merchants'][:3]:
                narrative.append(f"  - {merchant}: {data['count']} visits, €{data['total']:.2f} spent")
        
        # Spending trend
        if 'previous_week_expenses' in week_data:
            if week_data['total_expenses'] > week_data['previous_week_expenses'] * 1.2:
                narrative.append("Spending increased significantly compared to the previous week.")
            elif week_data['total_expenses'] < week_data['previous_week_expenses'] * 0.8:
                narrative.append("Spending decreased notably compared to the previous week.")
        
        return " ".join(narrative)
    
    def _generate_monthly_summary_narrative(self, month: str, year: int, transactions_df: pd.DataFrame) -> str:
        """Generate narrative summary for a month"""
        total_income = transactions_df[transactions_df['Importe'] > 0]['Importe'].sum()
        total_expenses = abs(transactions_df[transactions_df['Importe'] < 0]['Importe'].sum())
        net_savings = total_income - total_expenses
        transaction_count = len(transactions_df)
        
        narrative = [f"Monthly Financial Report for {month} {year}:"]
        narrative.append(f"This month recorded {transaction_count} financial transactions.")
        
        # Income analysis
        if total_income > 0:
            narrative.append(f"Total monthly income was €{total_income:.2f}.")
            income_sources = transactions_df[transactions_df['Importe'] > 0].groupby(
                transactions_df.apply(lambda x: self._categorize(x), axis=1)
            )['Importe'].sum().sort_values(ascending=False)
            
            if not income_sources.empty:
                narrative.append("Income sources included:")
                for source, amount in income_sources.items():
                    narrative.append(f"  - {source}: €{amount:.2f}")
        
        # Expense analysis
        if total_expenses > 0:
            narrative.append(f"Total monthly expenses were €{total_expenses:.2f}.")
            
            # Category breakdown
            expense_categories = transactions_df[transactions_df['Importe'] < 0].groupby(
                transactions_df[transactions_df['Importe'] < 0].apply(lambda x: self._categorize(x), axis=1)
            )['Importe'].apply(lambda x: abs(x.sum())).sort_values(ascending=False)
            
            if not expense_categories.empty:
                narrative.append("Major expense categories were:")
                for category, amount in expense_categories.head(5).items():
                    percentage = (amount / total_expenses) * 100
                    narrative.append(f"  - {category}: €{amount:.2f} ({percentage:.1f}% of total expenses)")
        
        # Savings analysis
        if net_savings > 0:
            savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
            narrative.append(f"The month ended with savings of €{net_savings:.2f}, a savings rate of {savings_rate:.1f}%.")
        else:
            narrative.append(f"The month ended with a deficit of €{abs(net_savings):.2f}.")
        
        # Financial health indicators
        if total_expenses > total_income * 1.1:
            narrative.append("Warning: Expenses exceeded income by more than 10% this month.")
        elif net_savings > total_income * 0.3:
            narrative.append("Excellent financial performance with over 30% of income saved.")
        
        # Spending patterns
        avg_daily_expense = total_expenses / 30
        narrative.append(f"Average daily spending was €{avg_daily_expense:.2f}.")
        
        # Most expensive day
        daily_expenses = transactions_df[transactions_df['Importe'] < 0].groupby('Fecha')['Importe'].sum()
        if not daily_expenses.empty:
            most_expensive_day = daily_expenses.idxmin()
            most_expensive_amount = abs(daily_expenses.min())
            narrative.append(f"The highest spending day was {most_expensive_day.strftime('%B %d')}, with €{most_expensive_amount:.2f} in expenses.")
        
        return " ".join(narrative)
    
    def generate_chunks(self):
        """Generate chunks using selected strategies"""
        logger.info(f"Generating chunks using strategies: {self.strategies}")
        
        for strategy in self.strategies:
            if strategy == 'narrative':
                self._generate_narrative_chunks()
            elif strategy == 'temporal':
                self._generate_temporal_chunks()
            elif strategy == 'categorical':
                self._generate_categorical_chunks()
            elif strategy == 'entity':
                self._generate_entity_chunks()
            elif strategy == 'pattern':
                self._generate_pattern_chunks()
        
        logger.info(f"Generated {len(self.chunks)} total chunks")
        self._print_chunk_statistics()
    
    def _generate_narrative_chunks(self):
        """Generate rich narrative chunks from transactions"""
        logger.info("Generating narrative chunks...")
        
        # Group by date for context
        self.df['Date'] = self.df['Fecha'].dt.date
        
        # Track context for narrative generation
        monthly_totals = defaultdict(lambda: defaultdict(float))
        merchant_visits = defaultdict(int)
        
        # 1. Generate individual transaction narratives
        for idx, row in self.df.iterrows():
            # Build context
            context = {
                'is_weekend': row['Fecha'].weekday() >= 5,
                'is_month_end': row['Fecha'].day >= 25,
                'merchant_visit_count': merchant_visits.get(self._extract_merchant(row['Concepto']), 0) + 1,
                'monthly_category_total': monthly_totals[row['Fecha'].month]
            }
            
            # Update tracking
            merchant = self._extract_merchant(row['Concepto'])
            if merchant:
                merchant_visits[merchant] += 1
            
            category = self._categorize(row)
            if row['Importe'] < 0:
                monthly_totals[row['Fecha'].month][category] += abs(row['Importe'])
            
            # Generate narrative
            narrative = self._generate_transaction_narrative(row, context)
            
            # Create chunk
            transaction_key = f"{idx}_{row['Fecha']}_{row['Importe']}"
            chunk = {
                'chunk_id': f"narrative_{hashlib.md5(transaction_key.encode()).hexdigest()[:12]}",
                'chunk_type': 'transaction_narrative',
                'date': row['Fecha'].strftime('%Y-%m-%d'),
                'amount': float(row['Importe']),
                'merchant': merchant,
                'category': category,
                'text': narrative,
                'metadata': {
                    'original_description': row['Concepto'],
                    'is_income': row['Importe'] > 0,
                    'weekday': row['Fecha'].strftime('%A'),
                    'month': row['Fecha'].strftime('%B %Y')
                }
            }
            
            self.chunks.append(chunk)
        
        # 2. Generate daily summaries
        for date, group in self.df.groupby('Date'):
            if len(group) > 1:  # Only create summary if multiple transactions
                transactions = group.to_dict('records')
                narrative = self._generate_daily_summary_narrative(pd.Timestamp(date), transactions)
                
                chunk = {
                    'chunk_id': f"daily_{hashlib.md5(str(date).encode()).hexdigest()[:12]}",
                    'chunk_type': 'daily_summary',
                    'date': str(date),
                    'transaction_count': len(group),
                    'text': narrative,
                    'metadata': {
                        'total_income': float(group[group['Importe'] > 0]['Importe'].sum()),
                        'total_expenses': float(abs(group[group['Importe'] < 0]['Importe'].sum())),
                        'net_flow': float(group['Importe'].sum())
                    }
                }
                
                self.chunks.append(chunk)
        
        # 3. Generate weekly summaries
        self.df['Week'] = self.df['Fecha'].dt.isocalendar().week
        self.df['Year'] = self.df['Fecha'].dt.year
        
        for (year, week), group in self.df.groupby(['Year', 'Week']):
            if len(group) > 3:  # Only create summary if enough transactions
                # Calculate week data
                merchant_data = defaultdict(lambda: {'count': 0, 'total': 0})
                for _, row in group.iterrows():
                    merchant = self._extract_merchant(row['Concepto'])
                    if merchant and row['Importe'] < 0:
                        merchant_data[merchant]['count'] += 1
                        merchant_data[merchant]['total'] += abs(row['Importe'])
                
                week_data = {
                    'week_num': int(week),
                    'year': int(year),
                    'start_date': group['Fecha'].min().strftime('%B %d'),
                    'end_date': group['Fecha'].max().strftime('%B %d'),
                    'transaction_count': len(group),
                    'total_income': float(group[group['Importe'] > 0]['Importe'].sum()) if not group[group['Importe'] > 0].empty else 0,
                    'total_expenses': float(abs(group[group['Importe'] < 0]['Importe'].sum())) if not group[group['Importe'] < 0].empty else 0,
                    'top_merchants': sorted(merchant_data.items(), key=lambda x: x[1]['total'], reverse=True)
                }
                
                narrative = self._generate_weekly_summary_narrative(week_data)
                
                chunk = {
                    'chunk_id': f"weekly_{year}_w{week:02d}",
                    'chunk_type': 'weekly_summary',
                    'week': f"{year}-W{week:02d}",
                    'text': narrative,
                    'metadata': week_data
                }
                
                self.chunks.append(chunk)
        
        # 4. Generate monthly summaries
        for (year, month), group in self.df.groupby([self.df['Fecha'].dt.year, self.df['Fecha'].dt.month]):
            month_name = datetime(year, month, 1).strftime('%B')
            narrative = self._generate_monthly_summary_narrative(month_name, year, group)
            
            chunk = {
                'chunk_id': f"monthly_{year}_{month:02d}",
                'chunk_type': 'monthly_summary',
                'month': f"{year}-{month:02d}",
                'text': narrative,
                'metadata': {
                    'transaction_count': len(group),
                    'total_income': float(group[group['Importe'] > 0]['Importe'].sum()),
                    'total_expenses': float(abs(group[group['Importe'] < 0]['Importe'].sum())),
                    'net_savings': float(group['Importe'].sum()),
                    'unique_merchants': len(set(self._extract_merchant(d) for d in group['Concepto'] if self._extract_merchant(d)))
                }
            }
            
            self.chunks.append(chunk)
        
        self.chunk_stats['narrative'] = len(self.chunks) - sum(self.chunk_stats.values())
    
    def _generate_temporal_chunks(self):
        """Generate time-based aggregation chunks"""
        logger.info("Generating temporal chunks...")
        base_count = len(self.chunks)
        
        # Hourly patterns (if time data available)
        # Daily patterns
        for date, group in self.df.groupby(self.df['Fecha'].dt.date):
            if len(group) < 2:
                continue
            
            chunk = self._create_temporal_chunk(date, group, 'daily')
            self.chunks.append(chunk)
        
        # Weekly patterns
        for (year, week), group in self.df.groupby(['Year', 'Week']):
            if len(group) < 3:
                continue
            
            chunk = self._create_temporal_chunk(f"{year}-W{week:02d}", group, 'weekly')
            self.chunks.append(chunk)
        
        # Monthly patterns
        for (year, month), group in self.df.groupby(['Year', 'Month']):
            chunk = self._create_temporal_chunk(f"{year}-{month:02d}", group, 'monthly')
            self.chunks.append(chunk)
        
        self.chunk_stats['temporal'] = len(self.chunks) - base_count
    
    def _generate_categorical_chunks(self):
        """Generate category-based chunks"""
        logger.info("Generating categorical chunks...")
        base_count = len(self.chunks)
        
        for category, profile in self.category_profiles.items():
            if profile['transaction_count'] < 2:
                continue
            
            # Get all transactions for this category
            cat_transactions = self.df[
                (self.df.apply(lambda x: self._categorize(x), axis=1) == category) &
                (~self.df['IsIncome'])
            ]
            
            if cat_transactions.empty:
                continue
            
            chunk = {
                'chunk_id': f"category_{hashlib.md5(category.encode()).hexdigest()[:8]}",
                'chunk_type': 'category_analysis',
                'category': category,
                'text': self._generate_category_narrative(category, profile, cat_transactions),
                'metadata': {
                    'total_spent': profile['total_spent'],
                    'transaction_count': profile['transaction_count'],
                    'unique_merchants': len(profile['merchants']),
                    'avg_transaction': profile['total_spent'] / profile['transaction_count'],
                    'peak_weekday': max(profile['weekly_pattern'].items(), 
                                       key=lambda x: x[1])[0] if profile['weekly_pattern'] else None
                }
            }
            
            self.chunks.append(chunk)
        
        self.chunk_stats['categorical'] = len(self.chunks) - base_count
    
    def _generate_entity_chunks(self):
        """Generate merchant/entity-based chunks"""
        logger.info("Generating entity chunks...")
        base_count = len(self.chunks)
        
        for merchant, profile in self.merchant_profiles.items():
            if profile['transaction_count'] < 2:
                continue
            
            chunk = {
                'chunk_id': f"entity_{hashlib.md5(merchant.encode()).hexdigest()[:8]}",
                'chunk_type': 'entity_profile',
                'entity': merchant,
                'text': self._generate_entity_narrative(merchant, profile),
                'metadata': {
                    'total_spent': profile['total_spent'],
                    'transaction_count': profile['transaction_count'],
                    'avg_transaction': profile['total_spent'] / profile['transaction_count'],
                    'categories': list(profile['categories']),
                    'is_recurring': profile['is_recurring'],
                    'recurrence_type': profile.get('recurrence_type')
                }
            }
            
            self.chunks.append(chunk)
        
        self.chunk_stats['entity'] = len(self.chunks) - base_count
    
    def _generate_pattern_chunks(self):
        """Generate pattern-based chunks (recurring, anomalies)"""
        logger.info("Generating pattern chunks...")
        base_count = len(self.chunks)
        
        # Recurring transactions chunk
        recurring_merchants = [
            (m, p) for m, p in self.merchant_profiles.items() 
            if p['is_recurring']
        ]
        
        if recurring_merchants:
            chunk = {
                'chunk_id': 'pattern_recurring',
                'chunk_type': 'pattern_analysis',
                'pattern_type': 'recurring',
                'text': self._generate_recurring_narrative(recurring_merchants),
                'metadata': {
                    'recurring_count': len(recurring_merchants),
                    'total_recurring_spend': sum(p['total_spent'] for _, p in recurring_merchants),
                    'merchants': [m for m, _ in recurring_merchants]
                }
            }
            self.chunks.append(chunk)
        
        # Anomaly detection chunk
        anomalies = self._detect_anomalies()
        if anomalies:
            chunk = {
                'chunk_id': 'pattern_anomalies',
                'chunk_type': 'pattern_analysis',
                'pattern_type': 'anomalies',
                'text': self._generate_anomaly_narrative(anomalies),
                'metadata': {
                    'anomaly_count': len(anomalies),
                    'types': list(set(a['type'] for a in anomalies))
                }
            }
            self.chunks.append(chunk)
        
        self.chunk_stats['pattern'] = len(self.chunks) - base_count
    
    def _create_temporal_chunk(self, period_id: Any, group: pd.DataFrame, period_type: str) -> Dict:
        """Create a temporal aggregation chunk"""
        total_income = group[group['IsIncome']]['Importe'].sum()
        total_expenses = abs(group[~group['IsIncome']]['Importe'].sum())
        
        # Category breakdown
        categories = {}
        for _, row in group[~group['IsIncome']].iterrows():
            cat = self._categorize(row)
            categories[cat] = categories.get(cat, 0) + abs(row['Importe'])
        
        return {
            'chunk_id': f"temporal_{period_type}_{hashlib.md5(str(period_id).encode()).hexdigest()[:8]}",
            'chunk_type': f'{period_type}_summary',
            'period': str(period_id),
            'text': self._generate_temporal_narrative(period_id, period_type, group),
            'metadata': {
                'period_type': period_type,
                'transaction_count': len(group),
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_flow': total_income - total_expenses,
                'categories': categories
            }
        }
    
    def _generate_temporal_narrative(self, period: Any, period_type: str, data: pd.DataFrame) -> str:
        """Generate narrative for temporal chunk"""
        total_in = data[data['IsIncome']]['Importe'].sum()
        total_out = abs(data[~data['IsIncome']]['Importe'].sum())
        net = total_in - total_out
        
        narrative = [f"Financial summary for {period_type} period {period}:"]
        narrative.append(f"There were {len(data)} transactions totaling €{total_in:.2f} "
                        f"in income and €{total_out:.2f} in expenses.")
        
        if net > 0:
            narrative.append(f"The period ended with a surplus of €{net:.2f}.")
        else:
            narrative.append(f"The period ended with a deficit of €{abs(net):.2f}.")
        
        # Top categories
        categories = {}
        for _, row in data[~data['IsIncome']].iterrows():
            cat = self._categorize(row)
            categories[cat] = categories.get(cat, 0) + abs(row['Importe'])
        
        if categories:
            top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            narrative.append(f"Top spending categories: " + 
                           ", ".join([f"{cat} (€{amt:.2f})" for cat, amt in top_cats]))
        
        return " ".join(narrative)
    
    def _generate_category_narrative(self, category: str, profile: Dict, transactions: pd.DataFrame) -> str:
        """Generate narrative for category chunk"""
        narrative = [f"Analysis of {category} spending:"]
        narrative.append(f"Total of {profile['transaction_count']} transactions "
                        f"amounting to €{profile['total_spent']:.2f}.")
        
        avg = profile['total_spent'] / profile['transaction_count']
        narrative.append(f"Average transaction: €{avg:.2f}.")
        
        if profile['merchants']:
            narrative.append(f"Includes purchases from {len(profile['merchants'])} "
                           f"different merchants: {', '.join(list(profile['merchants'])[:5])}.")
        
        # Temporal patterns
        if profile['weekly_pattern']:
            peak_day = max(profile['weekly_pattern'].items(), key=lambda x: x[1])
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            narrative.append(f"Peak spending day: {day_names[peak_day[0]]}.")
        
        return " ".join(narrative)
    
    def _generate_entity_narrative(self, merchant: str, profile: Dict) -> str:
        """Generate narrative for entity/merchant chunk"""
        narrative = [f"Spending profile for {merchant}:"]
        narrative.append(f"{profile['transaction_count']} transactions "
                        f"totaling €{profile['total_spent']:.2f}.")
        
        avg = profile['total_spent'] / profile['transaction_count']
        narrative.append(f"Average transaction: €{avg:.2f}.")
        
        if profile['categories']:
            narrative.append(f"Categories: {', '.join(profile['categories'])}.")
        
        if profile['is_recurring']:
            narrative.append(f"This is a {profile.get('recurrence_type', 'recurring')} payment.")
        
        # Spending trend
        if len(profile['typical_amount']) > 1:
            amounts = profile['typical_amount']
            if amounts[-1] > np.mean(amounts) * 1.2:
                narrative.append("Recent spending is above average.")
            elif amounts[-1] < np.mean(amounts) * 0.8:
                narrative.append("Recent spending is below average.")
        
        return " ".join(narrative)
    
    def _generate_recurring_narrative(self, recurring: List[Tuple[str, Dict]]) -> str:
        """Generate narrative for recurring transactions"""
        narrative = [f"Identified {len(recurring)} recurring payments:"]
        
        total_recurring = sum(p['total_spent'] for _, p in recurring)
        monthly_impact = total_recurring / len(self.df['Month'].unique())
        
        narrative.append(f"Total recurring expenses: €{total_recurring:.2f} "
                        f"(approximately €{monthly_impact:.2f} per month).")
        
        for merchant, profile in recurring[:5]:  # Top 5
            avg = profile['total_spent'] / profile['transaction_count']
            narrative.append(f"- {merchant}: {profile.get('recurrence_type', 'recurring')}, "
                           f"€{avg:.2f} per transaction")
        
        return " ".join(narrative)
    
    def _generate_anomaly_narrative(self, anomalies: List[Dict]) -> str:
        """Generate narrative for anomalies"""
        narrative = [f"Detected {len(anomalies)} unusual patterns:"]
        
        for anomaly in anomalies[:5]:  # Top 5
            if anomaly['type'] == 'high_amount':
                narrative.append(f"- Unusually high transaction: €{anomaly['amount']:.2f} "
                               f"at {anomaly['merchant']} on {anomaly['date']}")
            elif anomaly['type'] == 'unusual_time':
                narrative.append(f"- Unusual timing: Transaction at {anomaly['time']} "
                               f"on {anomaly['date']}")
            elif anomaly['type'] == 'new_merchant':
                narrative.append(f"- First transaction with {anomaly['merchant']} "
                               f"for €{anomaly['amount']:.2f}")
        
        return " ".join(narrative)
    
    def _detect_anomalies(self) -> List[Dict]:
        """Detect anomalous transactions"""
        anomalies = []
        
        # High amount anomalies (>3 std from mean)
        expenses = self.df[~self.df['IsIncome']]['Importe'].abs()
        if len(expenses) > 0:
            mean_expense = expenses.mean()
            std_expense = expenses.std()
            threshold = mean_expense + (3 * std_expense)
            
            for _, row in self.df[~self.df['IsIncome']].iterrows():
                if abs(row['Importe']) > threshold:
                    anomalies.append({
                        'type': 'high_amount',
                        'amount': abs(row['Importe']),
                        'merchant': self._extract_merchant(row['Concepto']),
                        'date': row['Fecha'].strftime('%Y-%m-%d'),
                        'description': row['Concepto']
                    })
        
        # New merchant detection
        seen_merchants = set()
        for _, row in self.df.iterrows():
            merchant = self._extract_merchant(row['Concepto'])
            if merchant and merchant not in seen_merchants:
                if abs(row['Importe']) > mean_expense * 2:  # First purchase is large
                    anomalies.append({
                        'type': 'new_merchant',
                        'merchant': merchant,
                        'amount': abs(row['Importe']),
                        'date': row['Fecha'].strftime('%Y-%m-%d')
                    })
                seen_merchants.add(merchant)
        
        return anomalies
    
    def _print_chunk_statistics(self):
        """Print statistics about generated chunks"""
        logger.info("\nChunk Statistics:")
        logger.info(f"  Total chunks: {len(self.chunks)}")
        
        # Count by type
        type_counts = defaultdict(int)
        for chunk in self.chunks:
            type_counts[chunk['chunk_type']] += 1
        
        for chunk_type, count in sorted(type_counts.items()):
            logger.info(f"  - {chunk_type}: {count}")
        
        # Strategy breakdown
        logger.info("\nBy strategy:")
        for strategy, count in self.chunk_stats.items():
            logger.info(f"  - {strategy}: {count}")
    
    def save_chunks(self, output_path: str):
        """Save chunks to JSON file"""
        # Prepare metadata
        type_counts = defaultdict(int)
        for chunk in self.chunks:
            type_counts[chunk['chunk_type']] += 1
        
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'strategies': self.strategies,
                'total_chunks': len(self.chunks),
                'anonymized': self.anonymized,
                'chunk_types': dict(type_counts),
                'strategy_counts': dict(self.chunk_stats),
                'data_period': {
                    'start': self.df['Fecha'].min().isoformat() if not self.df.empty else None,
                    'end': self.df['Fecha'].max().isoformat() if not self.df.empty else None,
                    'transaction_count': len(self.df)
                }
            },
            'chunks': self.chunks
        }
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n✅ Saved {len(self.chunks)} chunks to {output_path}")
        
        # Print sample chunks
        if self.chunks:
            logger.info("\nSample chunks:")
            for chunk in self.chunks[:2]:
                logger.info(f"\n--- {chunk['chunk_type']} ({chunk['chunk_id']}) ---")
                text_preview = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                logger.info(text_preview)


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate chunks for financial RAG')
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSON or CSV file with transactions'
    )
    parser.add_argument(
        '--output', '-o',
        default='output_chunks/adaptive_chunks.json',
        help='Output JSON file for chunks'
    )
    parser.add_argument(
        '--strategies', '-s',
        nargs='+',
        choices=AdaptiveChunkGenerator.CHUNK_STRATEGIES,
        default=None,
        help='Chunking strategies to use (default: all)'
    )
    parser.add_argument(
        '--anonymized',
        action='store_true',
        help='Flag if data is anonymized'
    )
    
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize generator
    if input_path.suffix.lower() == '.json':
        generator = AdaptiveChunkGenerator(
            json_path=str(input_path),
            anonymized=args.anonymized,
            strategies=args.strategies
        )
    elif input_path.suffix.lower() == '.csv':
        generator = AdaptiveChunkGenerator(
            csv_path=str(input_path),
            anonymized=args.anonymized,
            strategies=args.strategies
        )
    else:
        logger.error(f"Unsupported file type: {input_path.suffix}")
        sys.exit(1)
    
    try:
        # Process
        generator.load_data()
        generator.generate_chunks()
        generator.save_chunks(args.output)
        
        logger.info("\n✅ Chunking complete!")
        logger.info(f"Output saved to: {args.output}")
        logger.info("\nNext steps:")
        logger.info("1. Generate embeddings for chunks")
        logger.info("2. Build RAG index (LightRAG/GraphRAG)")
        logger.info("3. Query the knowledge graph")
        
    except Exception as e:
        logger.error(f"Error during chunking: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()