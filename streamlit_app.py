import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Mauritius Job Market Analysis",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["� Jobs With Salary", "❓ Jobs Without Salary"])

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
    .no-salary-highlight {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class JobDataAnalyzer:
    def __init__(self, df):
        """Initialize with DataFrame"""
        self.df = df
        self.clean_data()
        
    def clean_data(self):
        """Clean and preprocess the job data"""
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Extract salary from description first, then clean salary column
        self.df['salary_extracted'] = self.df['description'].apply(self.extract_salary_from_description)
        self.df['salary_cleaned'] = self.df['salary_extracted'].apply(self.clean_salary)
        
        # Extract numeric salary values
        self.df['salary_min'] = self.df['salary_cleaned'].apply(self.extract_salary_min)
        self.df['salary_max'] = self.df['salary_cleaned'].apply(self.extract_salary_max)
        self.df['salary_average'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        # Clean location data
        self.df['location_cleaned'] = self.df['description'].apply(self.extract_location)
        
        # Extract job type from description
        self.df['job_type_cleaned'] = self.df['description'].apply(self.extract_job_type)
        
        # Extract experience requirements
        self.df['experience_cleaned'] = self.df['description'].apply(self.extract_experience)
        
        # Extract skills from description
        self.df['skills'] = self.df['description'].apply(self.extract_skills)
        
        # Clean company names
        self.df['company_cleaned'] = self.df['company'].str.strip()
        
        # Convert posted_date to datetime
        self.df['posted_date_cleaned'] = self.df['description'].apply(self.extract_posted_date)
    
    def extract_salary_from_description(self, description):
        """Extract salary information from description text"""
        if pd.isna(description):
            return 'Not specified'
        
        desc = str(description)
        
        # Look for salary patterns in the description
        # Pattern 1: "Location SalaryRange JobType" format
        location_salary_pattern = r'([A-Za-z\s]+)\s+(Negotiable|Not disclosed|\d{1,2}(?:,\d{3})*(?:\s*-\s*\d{1,2}(?:,\d{3})*)?)\s+(Permanent|Contract|Temporary)'
        match = re.search(location_salary_pattern, desc)
        if match:
            salary = match.group(2)
            return salary.strip()
        
        # Pattern 2: Look for salary ranges with currency indicators or reasonable amounts
        salary_patterns = [
            r'MUR\s*(\d{1,2}(?:,\d{3})*)\s*-\s*(\d{1,2}(?:,\d{3})*)',
            r'(\d{1,2}(?:,\d{3})*)\s*-\s*(\d{1,2}(?:,\d{3})*)\s*MUR',
            r'Rs\s*(\d{1,2}(?:,\d{3})*)\s*-\s*(\d{1,2}(?:,\d{3})*)',
            r'(\d{1,2}(?:,\d{3})*)\s*-\s*(\d{1,2}(?:,\d{3})*)\s*Rs',
            r'MUR\s*(\d{1,2}(?:,\d{3})*)',
            r'Rs\s*(\d{1,2}(?:,\d{3})*)',
            r'(\d{1,2}(?:,\d{3})*)\s*MUR',
            r'(\d{1,2}(?:,\d{3})*)\s*Rs',
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, desc, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1] is not None:
                    return f'{groups[0]} - {groups[1]}'
                elif groups[0] is not None:
                    # Only return if it's a reasonable salary amount (>= 1000)
                    try:
                        salary_value = int(groups[0].replace(',', ''))
                        if salary_value >= 1000:
                            return groups[0]
                    except ValueError:
                        pass
        
        # Pattern 3: Look for "Negotiable"
        if 'Negotiable' in desc:
            return 'Negotiable'
        
        # Pattern 4: Look for "Not disclosed"
        if 'Not disclosed' in desc:
            return 'Not disclosed'
        
        return 'Not specified'
    
    def clean_salary(self, salary_text):
        """Clean salary text and standardize format"""
        if pd.isna(salary_text) or salary_text == 'Not specified':
            return 'Not specified'
        
        salary_text = str(salary_text).strip()
        
        # Handle "Negotiable" case
        if 'Negotiable' in salary_text or 'Not disclosed' in salary_text:
            return 'Negotiable'
        
        # Remove currency symbols and clean
        salary_text = re.sub(r'[MURRs]|MUR|Rs', '', salary_text)
        salary_text = re.sub(r',', '', salary_text)
        salary_text = re.sub(r'\s+', ' ', salary_text)
        
        return salary_text.strip()
    
    def extract_salary_min(self, salary_text):
        """Extract minimum salary value"""
        if pd.isna(salary_text) or 'Not specified' in str(salary_text) or 'Negotiable' in str(salary_text):
            return np.nan
        
        # Handle salary ranges with commas (e.g., "51,000 - 75,000")
        range_pattern = r'(\d{1,2}(?:,\d{3})*)\s*-\s*\d{1,2}(?:,\d{3})*'
        match = re.search(range_pattern, str(salary_text))
        if match:
            return float(match.group(1).replace(',', ''))
        
        # Handle single salary values with commas (e.g., "51,000")
        single_pattern = r'(\d{1,2}(?:,\d{3})*)'
        match = re.search(single_pattern, str(salary_text))
        if match:
            return float(match.group(1).replace(',', ''))
        
        return np.nan
    
    def extract_salary_max(self, salary_text):
        """Extract maximum salary value"""
        if pd.isna(salary_text) or 'Not specified' in str(salary_text) or 'Negotiable' in str(salary_text):
            return np.nan
        
        # Handle salary ranges with commas (e.g., "51,000 - 75,000")
        range_pattern = r'\d{1,2}(?:,\d{3})*\s*-\s*(\d{1,2}(?:,\d{3})*)'
        match = re.search(range_pattern, str(salary_text))
        if match:
            return float(match.group(1).replace(',', ''))
        
        # Handle single salary values with commas (e.g., "75,000")
        single_pattern = r'(\d{1,2}(?:,\d{3})*)'
        match = re.search(single_pattern, str(salary_text))
        if match:
            return float(match.group(1).replace(',', ''))
        
        return np.nan
    
    def extract_location(self, description):
        """Extract location from description"""
        if pd.isna(description):
            return 'Not specified'
        
        locations = ['Port Louis', 'Plaine Wilhems', 'Pamplemousses', 'Black River', 
                    'Moka', 'Flacq', 'Riviere du Rempart', 'Grand Port', 'Savanne',
                    'Mauritius', 'Rodrigues']
        
        desc_str = str(description).upper()
        for location in locations:
            if location.upper() in desc_str:
                return location
        
        return 'Not specified'
    
    def extract_job_type(self, description):
        """Extract job type from description"""
        if pd.isna(description):
            return 'Not specified'
        
        desc_str = str(description).upper()
        
        if 'PERMANENT' in desc_str or 'CDI' in desc_str:
            return 'Permanent'
        elif 'CONTRACT' in desc_str or 'TEMPORARY' in desc_str:
            return 'Contract/Temporary'
        elif 'TRAINEE' in desc_str or 'INTERNSHIP' in desc_str:
            return 'Trainee'
        elif 'PART-TIME' in desc_str:
            return 'Part-time'
        else:
            return 'Not specified'
    
    def extract_experience(self, description):
        """Extract experience requirements from description"""
        if pd.isna(description):
            return 'Not specified'
        
        desc_str = str(description)
        
        exp_patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\s*ans?',
            r'minimum\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, desc_str, re.IGNORECASE)
            if match:
                years = int(match.group(1))
                if years <= 2:
                    return 'Entry Level (0-2 years)'
                elif years <= 5:
                    return 'Mid Level (3-5 years)'
                elif years <= 10:
                    return 'Senior Level (6-10 years)'
                else:
                    return 'Executive Level (10+ years)'
        
        return 'Not specified'
    
    def extract_skills(self, description):
        """Extract skills from job description"""
        if pd.isna(description):
            return []
        
        desc_str = str(description).lower()
        
        skills_list = [
            'excel', 'word', 'powerpoint', 'microsoft office', 'communication',
            'teamwork', 'leadership', 'project management', 'customer service',
            'sales', 'marketing', 'accounting', 'finance', 'hr', 'recruitment',
            'french', 'english', 'bilingual', 'computer', 'software', 'it',
            'networking', 'programming', 'database', 'analysis', 'problem solving',
            'time management', 'organization', 'planning', 'research', 'writing'
        ]
        
        found_skills = []
        for skill in skills_list:
            if skill in desc_str:
                found_skills.append(skill.title())
        
        return found_skills
    
    def extract_posted_date(self, description):
        """Extract posted date from description"""
        if pd.isna(description):
            return None
        
        desc_str = str(description)
        date_pattern = r'Added\s+(\d{2}/\d{2}/\d{4})'
        match = re.search(date_pattern, desc_str)
        if match:
            return pd.to_datetime(match.group(1), format='%d/%m/%Y')
        
        return None

def load_data():
    """Load and prepare the data"""
    try:
        df = pd.read_csv('mauritius_jobs_1000.csv')
        analyzer = JobDataAnalyzer(df)
        return analyzer.df
    except FileNotFoundError:
        st.error("❌ File 'mauritius_jobs_1000.csv' not found. Please make sure the file is in the same directory.")
        return None

def create_overview_metrics(df):
    """Create overview metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Job Listings", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Companies", f"{df['company_cleaned'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Locations", f"{df['location_cleaned'].nunique()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        salary_data = df[df['salary_average'].notna()]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jobs with Salary Info", f"{len(salary_data)} ({len(salary_data)/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)

def create_job_type_analysis(df):
    """Create job type analysis section"""
    st.subheader("🏢 Job Type Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        job_type_counts = df['job_type_cleaned'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(job_type_counts.values, labels=job_type_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Job Type Distribution')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Bar chart with percentages
        job_type_pct = (job_type_counts / len(df) * 100).round(1)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(job_type_pct.index, job_type_pct.values)
        ax.set_title('Job Type Percentage')
        ax.set_ylabel('Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, pct in zip(bars, job_type_pct.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**Key Insights:**")
    permanent_pct = (job_type_counts.get('Permanent', 0) / len(df) * 100)
    st.write(f"• **{permanent_pct:.1f}%** of jobs are permanent positions, indicating job market stability")
    st.write(f"• Only **{(job_type_counts.get('Contract/Temporary', 0) / len(df) * 100):.1f}%** are contract/temporary roles")
    st.markdown('</div>', unsafe_allow_html=True)

def create_location_analysis(df):
    """Create location analysis section"""
    st.subheader("📍 Geographic Distribution")
    
    # Top locations
    location_counts = df['location_cleaned'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(location_counts.index, location_counts.values)
        ax.set_title('Top 10 Locations by Job Count')
        ax.set_xlabel('Number of Jobs')
        
        # Add value labels
        for bar, count in zip(bars, location_counts.values):
            width = bar.get_width()
            ax.text(width + max(location_counts.values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{count}', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Location percentages
        location_pct = (location_counts / len(df) * 100).round(1)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(location_pct.index, location_pct.values)
        ax.set_title('Top 10 Locations by Percentage')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Location')
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for bar, pct in zip(bars, location_pct.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Location insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**Geographic Insights:**")
    top_3_locations = location_counts.head(3)
    total_top_3 = top_3_locations.sum()
    st.write(f"• Top 3 locations account for **{(total_top_3/len(df)*100):.1f}%** of all jobs")
    st.write(f"• **{location_counts.index[0]}** leads with **{location_counts.iloc[0]}** job postings")
    st.markdown('</div>', unsafe_allow_html=True)

def create_salary_analysis(df):
    """Create comprehensive salary analysis section"""
    st.subheader("💰 Salary Analysis")
    
    # Overall salary statistics
    salary_data = df[df['salary_average'].notna()]
    negotiable_count = df[df['salary_cleaned'] == 'Negotiable'].shape[0]
    not_specified_count = df[df['salary_cleaned'] == 'Not specified'].shape[0]
    
    # Salary overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jobs with Salary Info", f"{len(salary_data)} ({len(salary_data)/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Negotiable Salaries", f"{negotiable_count} ({negotiable_count/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if len(salary_data) > 0:
            avg_salary = salary_data['salary_average'].mean()
            st.metric("Average Salary", f"MUR {avg_salary:,.0f}")
        else:
            st.metric("Average Salary", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if len(salary_data) > 0:
            median_salary = salary_data['salary_average'].median()
            st.metric("Median Salary", f"MUR {median_salary:,.0f}")
        else:
            st.metric("Median Salary", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(salary_data) == 0:
        st.warning("No numeric salary data available for detailed analysis")
        return
    
    # Detailed salary analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(salary_data['salary_average'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax.set_title('Salary Distribution (MUR/month)')
        ax.set_xlabel('Salary (MUR)')
        ax.set_ylabel('Frequency')
        ax.axvline(salary_data['salary_average'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {salary_data["salary_average"].mean():,.0f}')
        ax.axvline(salary_data['salary_average'].median(), color='green', linestyle='--', 
                  label=f'Median: {salary_data["salary_average"].median():,.0f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Salary range analysis (min vs max)
        range_data = df[(df['salary_min'].notna()) & (df['salary_max'].notna())]
        if len(range_data) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(range_data['salary_min'], range_data['salary_max'], alpha=0.6, color='coral')
            ax.plot([range_data['salary_min'].min(), range_data['salary_max'].max()], 
                   [range_data['salary_min'].min(), range_data['salary_max'].max()], 
                   'r--', alpha=0.8, label='Equal Min/Max')
            ax.set_title('Salary Range Analysis')
            ax.set_xlabel('Minimum Salary (MUR)')
            ax.set_ylabel('Maximum Salary (MUR)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No salary range data available")
    
    # Salary statistics table
    st.markdown("### Detailed Salary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        stats_data = {
            'Mean': salary_data['salary_average'].mean(),
            'Median': salary_data['salary_average'].median(),
            'Min': salary_data['salary_average'].min(),
            'Max': salary_data['salary_average'].max(),
            'Std Dev': salary_data['salary_average'].std(),
            '25th Percentile': salary_data['salary_average'].quantile(0.25),
            '75th Percentile': salary_data['salary_average'].quantile(0.75)
        }
        
        for stat, value in stats_data.items():
            st.metric(stat, f"MUR {value:,.0f}")
    
    with col2:
        # Salary brackets analysis
        salary_brackets = pd.cut(salary_data['salary_average'], 
                                bins=[0, 15000, 25000, 35000, 50000, float('inf')],
                                labels=['< 15K', '15K-25K', '25K-35K', '35K-50K', '> 50K'])
        bracket_counts = salary_brackets.value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(bracket_counts.index, bracket_counts.values, color='lightgreen')
        ax.set_title('Salary Distribution by Brackets')
        ax.set_xlabel('Salary Range (MUR)')
        ax.set_ylabel('Number of Jobs')
        ax.set_xticklabels(bracket_counts.index, rotation=45, ha='right')
        
        # Add percentage labels
        for bar, count in zip(bars, bracket_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(bracket_counts.values)*0.01,
                    f'{count}\n({count/len(salary_data)*100:.1f}%)', 
                    ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Salary by job type
    if salary_data['job_type_cleaned'].nunique() > 1:
        st.markdown("### Average Salary by Job Type")
        salary_by_type = salary_data.groupby('job_type_cleaned')['salary_average'].agg(['mean', 'count', 'std']).round(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(salary_by_type.index, salary_by_type['mean'], yerr=salary_by_type['std'], 
                     capsize=5, alpha=0.7, color='orange')
        ax.set_title('Average Salary by Job Type (with std deviation)')
        ax.set_ylabel('Average Salary (MUR)')
        ax.set_xlabel('Job Type')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels and sample size
        for i, (bar, (avg, count)) in enumerate(zip(bars, zip(salary_by_type['mean'], salary_by_type['count']))):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(salary_by_type['mean'])*0.02,
                    f'{avg:,.0f}\n(n={count})', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Salary by location
    if salary_data['location_cleaned'].nunique() > 1:
        st.markdown("### Average Salary by Location")
        salary_by_location = salary_data.groupby('location_cleaned')['salary_average'].agg(['mean', 'count']).round(0)
        salary_by_location = salary_by_location[salary_by_location['count'] >= 3]  # Filter for locations with at least 3 jobs
        salary_by_location = salary_by_location.sort_values('mean', ascending=False).head(10)
        
        if len(salary_by_location) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(salary_by_location.index, salary_by_location['mean'], color='purple')
            ax.set_title('Top 10 Locations by Average Salary')
            ax.set_xlabel('Average Salary (MUR)')
            
            # Add value labels and sample size
            for bar, (avg, count) in zip(bars, zip(salary_by_location['mean'], salary_by_location['count'])):
                width = bar.get_width()
                ax.text(width + max(salary_by_location['mean'])*0.01, bar.get_y() + bar.get_height()/2.,
                        f'{avg:,.0f}\n(n={count})', ha='left', va='center')
            
            st.pyplot(fig)
            plt.close()
    
    # Salary insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**Salary Insights:**")
    st.write(f"• **{len(salary_data)}** jobs have specific salary ranges ({len(salary_data)/len(df)*100:.1f}% of total)")
    st.write(f"• **{negotiable_count}** jobs have negotiable salaries ({negotiable_count/len(df)*100:.1f}% of total)")
    if len(salary_data) > 0:
        st.write(f"• Salary range: **MUR {salary_data['salary_average'].min():,.0f}** - **MUR {salary_data['salary_average'].max():,.0f}**")
        st.write(f"• **{len(salary_data[salary_data['salary_average'] > 35000])}** jobs pay above MUR 35,000 ({len(salary_data[salary_data['salary_average'] > 35000])/len(salary_data)*100:.1f}% of salaried jobs)")
    st.markdown('</div>', unsafe_allow_html=True)

def create_high_salary_analysis(df):
    """Create analysis for jobs with salaries higher than 10,000 MUR"""
    st.subheader("🚀 High-Paying Jobs Analysis (Salary > MUR 10,000)")
    
    # Filter jobs with salary > 10,000 MUR
    high_salary_df = df[df['salary_average'] > 10000].copy()
    
    if len(high_salary_df) == 0:
        st.warning("No jobs with salaries above MUR 10,000 found in the dataset.")
        return
    
    # Overview metrics for high-paying jobs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jobs > MUR 10,000", f"{len(high_salary_df)} ({len(high_salary_df)/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_high_salary = high_salary_df['salary_average'].mean()
        st.metric("Avg High Salary", f"MUR {avg_high_salary:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        max_salary = high_salary_df['salary_average'].max()
        st.metric("Highest Salary", f"MUR {max_salary:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        unique_companies = high_salary_df['company_cleaned'].nunique()
        st.metric("Companies", f"{unique_companies}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Salary distribution for high-paying jobs
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(high_salary_df['salary_average'], bins=15, alpha=0.7, edgecolor='black', color='gold')
        ax.set_title('High-Paying Jobs Distribution')
        ax.set_xlabel('Salary (MUR)')
        ax.set_ylabel('Frequency')
        ax.axvline(high_salary_df['salary_average'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {high_salary_df["salary_average"].mean():,.0f}')
        ax.axvline(high_salary_df['salary_average'].median(), color='green', linestyle='--', 
                  label=f'Median: {high_salary_df["salary_average"].median():,.0f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Job type distribution for high-paying jobs
        job_type_counts = high_salary_df['job_type_cleaned'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(job_type_counts.index, job_type_counts.values, color='lightcoral')
        ax.set_title('Job Types for High-Paying Jobs')
        ax.set_ylabel('Number of Jobs')
        ax.set_xlabel('Job Type')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, job_type_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(job_type_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Top locations for high-paying jobs
    st.markdown("### Top Locations for High-Paying Jobs")
    location_counts = high_salary_df['location_cleaned'].value_counts().head(10)
    
    if len(location_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(location_counts.index, location_counts.values, color='lightgreen')
        ax.set_title('Top 10 Locations for High-Paying Jobs')
        ax.set_xlabel('Number of Jobs')
        
        # Add value labels
        for bar, count in zip(bars, location_counts.values):
            width = bar.get_width()
            ax.text(width + max(location_counts.values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{count}', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    # Average salary by job type for high-paying jobs
    st.markdown("### Average Salary by Job Type (High-Paying Jobs)")
    salary_by_type = high_salary_df.groupby('job_type_cleaned')['salary_average'].agg(['mean', 'count']).round(0)
    salary_by_type = salary_by_type[salary_by_type['count'] >= 2]  # Filter for at least 2 jobs
    salary_by_type = salary_by_type.sort_values('mean', ascending=False)
    
    if len(salary_by_type) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(salary_by_type.index, salary_by_type['mean'], color='purple')
        ax.set_title('Average Salary by Job Type (High-Paying Jobs)')
        ax.set_xlabel('Average Salary (MUR)')
        
        # Add value labels and sample size
        for bar, (avg, count) in zip(bars, zip(salary_by_type['mean'], salary_by_type['count'])):
            width = bar.get_width()
            ax.text(width + max(salary_by_type['mean'])*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{avg:,.0f}\n(n={count})', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    # Data table for high-paying jobs
    st.markdown("### High-Paying Jobs List")
    display_df = high_salary_df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                               'experience_cleaned', 'salary_min', 'salary_max', 'salary_average']].copy()
    display_df.columns = ['Title', 'Company', 'Location', 'Job Type', 'Experience', 'Min Salary', 'Max Salary', 'Avg Salary']
    
    # Format salary columns
    display_df['Min Salary'] = display_df['Min Salary'].fillna('').apply(lambda x: f"MUR {x:,.0f}" if x != '' else '')
    display_df['Max Salary'] = display_df['Max Salary'].fillna('').apply(lambda x: f"MUR {x:,.0f}" if x != '' else '')
    display_df['Avg Salary'] = display_df['Avg Salary'].apply(lambda x: f"MUR {x:,.0f}")
    
    st.dataframe(display_df.head(50), use_container_width=True)
    
    # Key insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**High-Paying Jobs Insights:**")
    st.write(f"• **{len(high_salary_df)}** jobs pay above MUR 10,000 ({len(high_salary_df)/len(df)*100:.1f}% of total jobs)")
    st.write(f"• Average salary for high-paying jobs: **MUR {avg_high_salary:,.0f}**")
    st.write(f"• **{unique_companies}** companies offer high-paying positions")
    
    if len(job_type_counts) > 0:
        top_job_type = job_type_counts.index[0]
        st.write(f"• **{top_job_type}** roles dominate high-paying jobs ({job_type_counts.iloc[0]} positions)")
    
    if len(location_counts) > 0:
        top_location = location_counts.index[0]
        st.write(f"• **{top_location}** has the most high-paying opportunities ({location_counts.iloc[0]} jobs)")
    
    st.write("• These positions likely represent senior, specialized, or high-demand roles in the Mauritian job market")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button for high-paying jobs
    csv_high_salary = high_salary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download High-Paying Jobs Dataset",
        data=csv_high_salary,
        file_name='mauritius_jobs_high_salary.csv',
        mime='text/csv'
    )

def create_experience_analysis(df):
    """Create experience level analysis section"""
    st.subheader("📈 Experience Requirements")
    
    exp_counts = df['experience_cleaned'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Experience level distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(exp_counts.index, exp_counts.values)
        ax.set_title('Experience Level Distribution')
        ax.set_ylabel('Number of Jobs')
        ax.set_xlabel('Experience Level')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, exp_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(exp_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Experience level percentages
        exp_pct = (exp_counts / len(df) * 100).round(1)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        ax.pie(exp_pct.values, labels=exp_pct.index, autopct='%1.1f%%', colors=colors[:len(exp_pct)])
        ax.set_title('Experience Level Percentage')
        st.pyplot(fig)
        plt.close()
    
    # Experience insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**Experience Insights:**")
    unspecified_pct = exp_pct.get('Not specified', 0)
    st.write(f"• **{unspecified_pct:.1f}%** of jobs don't specify experience requirements")
    entry_level_pct = exp_pct.get('Entry Level (0-2 years)', 0)
    st.write(f"• **{entry_level_pct:.1f}%** are entry-level positions")
    executive_pct = exp_pct.get('Executive Level (10+ years)', 0)
    st.write(f"• **{executive_pct:.1f}%** require executive-level experience")
    st.markdown('</div>', unsafe_allow_html=True)

def create_skills_with_salary_analysis(df):
    """Create skills analysis with expected salary insights"""
    st.subheader("🎯 Skills Analysis with Expected Salary")
    
    # Extract all skills from the dataset
    all_skills = []
    for skills_list in df['skills']:
        if isinstance(skills_list, list):
            all_skills.extend(skills_list)
    
    if not all_skills:
        st.warning("No skills data available for analysis.")
        return
    
    # Create skill frequency dataframe
    skill_counts = pd.Series(all_skills).value_counts().head(20)
    
    # Calculate average salary by skill
    skill_salary_data = []
    for skill in skill_counts.index:
        skill_jobs = df[df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
        if len(skill_jobs) > 0 and skill_jobs['salary_average'].notna().sum() > 0:
            avg_salary = skill_jobs['salary_average'].mean()
            min_salary = skill_jobs['salary_average'].min()
            max_salary = skill_jobs['salary_average'].max()
            job_count = len(skill_jobs)
            
            skill_salary_data.append({
                'Skill': skill,
                'Job Count': job_count,
                'Average Salary': avg_salary,
                'Min Salary': min_salary,
                'Max Salary': max_salary
            })
    
    if skill_salary_data:
        skill_salary_df = pd.DataFrame(skill_salary_data).sort_values('Average Salary', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top paying skills
            st.write("**💰 Top 10 Highest Paying Skills:**")
            top_paying_skills = skill_salary_df.head(10)
            top_paying_skills_display = top_paying_skills.copy()
            top_paying_skills_display['Average Salary'] = top_paying_skills_display['Average Salary'].apply(lambda x: f'MUR {x:,.0f}')
            top_paying_skills_display['Min Salary'] = top_paying_skills_display['Min Salary'].apply(lambda x: f'MUR {x:,.0f}')
            top_paying_skills_display['Max Salary'] = top_paying_skills_display['Max Salary'].apply(lambda x: f'MUR {x:,.0f}')
            st.dataframe(top_paying_skills_display[['Skill', 'Job Count', 'Average Salary', 'Min Salary', 'Max Salary']], 
                       use_container_width=True)
        
        with col2:
            # Most in-demand skills
            st.write("**🔥 Top 10 Most In-Demand Skills:**")
            most_demanded = skill_counts.head(10)
            most_demanded_df = pd.DataFrame({
                'Skill': most_demanded.index,
                'Job Count': most_demanded.values
            })
            st.dataframe(most_demanded_df, use_container_width=True)
        
        # Skills salary distribution chart
        st.write("**📊 Salary Distribution by Top Skills:**")
        fig, ax = plt.subplots(figsize=(12, 8))
        top_skills_for_chart = skill_salary_df.head(15)
        
        # Create box plot data
        skill_box_data = []
        skill_labels = []
        for _, row in top_skills_for_chart.iterrows():
            skill = row['Skill']
            skill_jobs = df[df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
            salaries = skill_jobs['salary_average'].dropna().tolist()
            if salaries:
                skill_box_data.append(salaries)
                skill_labels.append(f"{skill}\n({row['Job Count']} jobs)")
        
        if skill_box_data:
            ax.boxplot(skill_box_data, labels=skill_labels)
            ax.set_title('Salary Distribution by Top Skills')
            ax.set_ylabel('Salary (MUR)')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Skills insights
        st.markdown("### 💡 Skills & Salary Insights")
        highest_paying_skill = skill_salary_df.iloc[0]
        most_demanded_skill = skill_counts.index[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>💰 Highest Paying Skill</h4>
                <p><strong>{highest_paying_skill['Skill']}</strong> with an average salary of MUR {highest_paying_skill['Average Salary']:,.0f}</p>
                <p>Range: MUR {highest_paying_skill['Min Salary']:,.0f} - MUR {highest_paying_skill['Max Salary']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🔥 Most In-Demand Skill</h4>
                <p><strong>{most_demanded_skill}</strong> appearing in {skill_counts[most_demanded_skill]} job postings</p>
                <p>This skill is in high demand across the Mauritian job market</p>
            </div>
            """, unsafe_allow_html=True)

def create_skills_with_expected_salary_analysis(df, salary_type):
    """Create skills analysis with expected salary ranges for negotiable/not disclosed jobs"""
    st.subheader(f"🎯 Skills Analysis for {salary_type} Jobs")
    
    # Extract all skills from the dataset
    all_skills = []
    for skills_list in df['skills']:
        if isinstance(skills_list, list):
            all_skills.extend(skills_list)
    
    if not all_skills:
        st.warning("No skills data available for analysis.")
        return
    
    # Create skill frequency dataframe
    skill_counts = pd.Series(all_skills).value_counts().head(20)
    
    # Get expected salary ranges from similar jobs with numeric salaries
    def get_expected_salary_range(skill):
        # This would ideally use a reference dataset or ML model
        # For now, we'll use industry averages based on job titles
        skill_jobs = df[df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
        
        # Estimate based on job titles and experience requirements
        senior_roles = ['Manager', 'Senior', 'Lead', 'Head', 'Director', 'Chief']
        mid_roles = ['Engineer', 'Specialist', 'Officer', 'Coordinator', 'Analyst']
        
        estimated_ranges = {
            'Entry Level': (15000, 25000),
            'Mid Level': (25000, 45000),
            'Senior Level': (45000, 80000),
            'Management': (60000, 120000)
        }
        
        # Simple heuristic based on job titles
        job_titles = skill_jobs['title'].str.lower().str.cat(sep=' ')
        
        if any(role.lower() in job_titles for role in senior_roles):
            return estimated_ranges['Management']
        elif any(role.lower() in job_titles for role in mid_roles):
            return estimated_ranges['Mid Level']
        else:
            return estimated_ranges['Entry Level']
    
    # Create skill analysis with expected ranges
    skill_analysis_data = []
    for skill in skill_counts.index:
        skill_jobs = df[df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
        job_count = len(skill_jobs)
        
        # Get expected salary range
        min_expected, max_expected = get_expected_salary_range(skill)
        avg_expected = (min_expected + max_expected) / 2
        
        skill_analysis_data.append({
            'Skill': skill,
            'Job Count': job_count,
            'Expected Min Salary': min_expected,
            'Expected Max Salary': max_expected,
            'Expected Average': avg_expected
        })
    
    skill_analysis_df = pd.DataFrame(skill_analysis_data).sort_values('Expected Average', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Skills with highest expected salaries
        st.write(f"**💰 Top 10 Skills - Highest Expected Salary ({salary_type}):**")
        top_expected = skill_analysis_df.head(10)
        top_expected_display = top_expected.copy()
        top_expected_display['Expected Average'] = top_expected_display['Expected Average'].apply(lambda x: f'MUR {x:,.0f}')
        top_expected_display['Expected Min Salary'] = top_expected_display['Expected Min Salary'].apply(lambda x: f'MUR {x:,.0f}')
        top_expected_display['Expected Max Salary'] = top_expected_display['Expected Max Salary'].apply(lambda x: f'MUR {x:,.0f}')
        st.dataframe(top_expected_display[['Skill', 'Job Count', 'Expected Average', 'Expected Min Salary', 'Expected Max Salary']], 
                   use_container_width=True)
    
    with col2:
        # Most in-demand skills
        st.write(f"**🔥 Top 10 Most In-Demand Skills ({salary_type}):**")
        most_demanded = skill_counts.head(10)
        most_demanded_df = pd.DataFrame({
            'Skill': most_demanded.index,
            'Job Count': most_demanded.values
        })
        st.dataframe(most_demanded_df, use_container_width=True)
    
    # Expected salary ranges visualization
    st.write(f"**📊 Expected Salary Ranges by Top Skills ({salary_type}):**")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_skills_for_chart = skill_analysis_df.head(12)
    
    y_pos = range(len(top_skills_for_chart))
    ax.barh(y_pos, top_skills_for_chart['Expected Average'], 
            xerr=[top_skills_for_chart['Expected Average'] - top_skills_for_chart['Expected Min Salary'],
                   top_skills_for_chart['Expected Max Salary'] - top_skills_for_chart['Expected Average']],
            capsize=5, alpha=0.7, color='skyblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{skill} ({count})" for skill, count in zip(top_skills_for_chart['Skill'], top_skills_for_chart['Job Count'])])
    ax.set_xlabel('Expected Average Salary (MUR)')
    ax.set_title(f'Expected Salary Ranges by Top Skills - {salary_type} Jobs')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Insights
    st.markdown(f"### 💡 {salary_type} Jobs - Skills & Salary Insights")
    highest_expected_skill = skill_analysis_df.iloc[0]
    most_demanded_skill = skill_counts.index[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>💰 Highest Expected Salary Skill</h4>
            <p><strong>{highest_expected_skill['Skill']}</strong> with expected average of MUR {highest_expected_skill['Expected Average']:,.0f}</p>
            <p>Expected range: MUR {highest_expected_skill['Expected Min Salary']:,.0f} - MUR {highest_expected_skill['Expected Max Salary']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <h4>🔥 Most In-Demand Skill</h4>
            <p><strong>{most_demanded_skill}</strong> appearing in {skill_counts[most_demanded_skill]} {salary_type.lower()} job postings</p>
            <p>This skill is highly valued in {salary_type.lower()} compensation roles</p>
        </div>
        """, unsafe_allow_html=True)

def create_negotiable_data_table(df):
    """Create data table for negotiable jobs"""
    st.subheader("📊 Negotiable Salary Jobs Data")
    
    # Show sample of negotiable jobs
    st.write("Sample of jobs with negotiable salary:")
    display_df = df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                     'experience_cleaned', 'skills']].copy()
    display_df.columns = ['Title', 'Company', 'Location', 'Job Type', 'Experience', 'Skills']
    
    # Format skills for better display
    display_df['Skills'] = display_df['Skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    
    st.dataframe(display_df.head(100), use_container_width=True)
    
    # Summary tables for negotiable jobs
    st.markdown("### Summary by Category (Negotiable Jobs)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job type summary for negotiable jobs
        job_type_counts = df['job_type_cleaned'].value_counts()
        st.write("**Job Types (Negotiable Jobs):**")
        job_type_df = pd.DataFrame({
            'Count': job_type_counts,
            'Percentage': (job_type_counts / len(df) * 100).round(1)
        })
        st.dataframe(job_type_df, use_container_width=True)
    
    with col2:
        # Location summary for negotiable jobs
        location_counts = df['location_cleaned'].value_counts().head(10)
        st.write("**Top 10 Locations (Negotiable Jobs):**")
        location_df = pd.DataFrame({
            'Count': location_counts,
            'Percentage': (location_counts / len(df) * 100).round(1)
        })
        st.dataframe(location_df, use_container_width=True)
    
    # Download button for negotiable jobs
    csv_negotiable = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Negotiable Jobs Dataset",
        data=csv_negotiable,
        file_name='mauritius_jobs_negotiable.csv',
        mime='text/csv'
    )

def create_not_disclosed_data_table(df):
    """Create data table for not disclosed jobs"""
    st.subheader("📊 Not Disclosed Salary Jobs Data")
    
    # Show sample of not disclosed jobs
    st.write("Sample of jobs with not disclosed salary:")
    display_df = df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                     'experience_cleaned', 'skills']].copy()
    display_df.columns = ['Title', 'Company', 'Location', 'Job Type', 'Experience', 'Skills']
    
    # Format skills for better display
    display_df['Skills'] = display_df['Skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    
    st.dataframe(display_df.head(100), use_container_width=True)
    
    # Summary tables for not disclosed jobs
    st.markdown("### Summary by Category (Not Disclosed Jobs)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job type summary for not disclosed jobs
        job_type_counts = df['job_type_cleaned'].value_counts()
        st.write("**Job Types (Not Disclosed Jobs):**")
        job_type_df = pd.DataFrame({
            'Count': job_type_counts,
            'Percentage': (job_type_counts / len(df) * 100).round(1)
        })
        st.dataframe(job_type_df, use_container_width=True)
    
    with col2:
        # Location summary for not disclosed jobs
        location_counts = df['location_cleaned'].value_counts().head(10)
        st.write("**Top 10 Locations (Not Disclosed Jobs):**")
        location_df = pd.DataFrame({
            'Count': location_counts,
            'Percentage': (location_counts / len(df) * 100).round(1)
        })
        st.dataframe(location_df, use_container_width=True)
    
    # Download button for not disclosed jobs
    csv_not_disclosed = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Not Disclosed Jobs Dataset",
        data=csv_not_disclosed,
        file_name='mauritius_jobs_not_disclosed.csv',
        mime='text/csv'
    )

def create_comprehensive_skills_analysis(df):
    """Create comprehensive skills analysis for all salary types"""
    st.subheader("🎯 Skills Analysis with Salary Insights")
    
    # Extract all skills from the dataset
    all_skills = []
    for skills_list in df['skills']:
        if isinstance(skills_list, list):
            all_skills.extend(skills_list)
    
    if not all_skills:
        st.warning("No skills data available for analysis.")
        return
    
    # Create skill frequency dataframe
    skill_counts = pd.Series(all_skills).value_counts().head(20)
    
    # Separate jobs with numeric salaries and those without
    numeric_salary_df = df[df['salary_average'].notna()]
    non_numeric_df = df[df['salary_average'].isna()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most in-demand skills overall
        st.write("**🔥 Top 10 Most In-Demand Skills:**")
        most_demanded_df = pd.DataFrame({
            'Skill': skill_counts.index,
            'Job Count': skill_counts.values
        })
        st.dataframe(most_demanded_df, use_container_width=True)
    
    with col2:
        # Skills with actual salary data (if available)
        if len(numeric_salary_df) > 0:
            st.write("**💰 Top Skills with Actual Salary Data:**")
            skill_salary_data = []
            for skill in skill_counts.index:
                skill_jobs = numeric_salary_df[numeric_salary_df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
                if len(skill_jobs) > 0:
                    avg_salary = skill_jobs['salary_average'].mean()
                    job_count = len(skill_jobs)
                    skill_salary_data.append({
                        'Skill': skill,
                        'Job Count': job_count,
                        'Average Salary': avg_salary
                    })
            
            if skill_salary_data:
                skill_salary_df = pd.DataFrame(skill_salary_data).sort_values('Average Salary', ascending=False).head(10)
                skill_salary_df['Average Salary'] = skill_salary_df['Average Salary'].apply(lambda x: f'MUR {x:,.0f}')
                st.dataframe(skill_salary_df[['Skill', 'Job Count', 'Average Salary']], use_container_width=True)
        else:
            st.write("**📊 Skills Distribution:**")
            st.write("No jobs with numeric salary data in current filter.")
    
    # Skills salary analysis section
    if len(numeric_salary_df) > 0:
        st.write("**💰 Skills with Actual Salary Analysis:**")
        
        # Calculate average salary by skill
        skill_salary_data = []
        for skill in skill_counts.index:
            skill_jobs = numeric_salary_df[numeric_salary_df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
            if len(skill_jobs) > 0:
                avg_salary = skill_jobs['salary_average'].mean()
                min_salary = skill_jobs['salary_average'].min()
                max_salary = skill_jobs['salary_average'].max()
                job_count = len(skill_jobs)
                
                skill_salary_data.append({
                    'Skill': skill,
                    'Job Count': job_count,
                    'Average Salary': avg_salary,
                    'Min Salary': min_salary,
                    'Max Salary': max_salary
                })
        
        if skill_salary_data:
            skill_salary_df = pd.DataFrame(skill_salary_data).sort_values('Average Salary', ascending=False)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            top_skills_for_chart = skill_salary_df.head(12)
            
            # Create box plot data
            skill_box_data = []
            skill_labels = []
            for _, row in top_skills_for_chart.iterrows():
                skill = row['Skill']
                skill_jobs = numeric_salary_df[numeric_salary_df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
                salaries = skill_jobs['salary_average'].dropna().tolist()
                if salaries:
                    skill_box_data.append(salaries)
                    skill_labels.append(f"{skill}\n({row['Job Count']} jobs)")
            
            if skill_box_data:
                ax.boxplot(skill_box_data, labels=skill_labels)
                ax.set_title('Salary Distribution by Top Skills (Actual Data)')
                ax.set_ylabel('Salary (MUR)')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # Expected salary analysis for non-numeric jobs
    if len(non_numeric_df) > 0:
        st.write("**🎯 Expected Salary Analysis for Non-Numeric Jobs:**")
        
        # Simple expected salary estimation
        def get_expected_salary_range(skill):
            skill_jobs = df[df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
            
            # Estimate based on job titles and experience requirements
            senior_roles = ['Manager', 'Senior', 'Lead', 'Head', 'Director', 'Chief']
            mid_roles = ['Engineer', 'Specialist', 'Officer', 'Coordinator', 'Analyst']
            
            estimated_ranges = {
                'Entry Level': (15000, 25000),
                'Mid Level': (25000, 45000),
                'Senior Level': (45000, 80000),
                'Management': (60000, 120000)
            }
            
            # Simple heuristic based on job titles
            job_titles = skill_jobs['title'].str.lower().str.cat(sep=' ')
            
            if any(role.lower() in job_titles for role in senior_roles):
                return estimated_ranges['Management']
            elif any(role.lower() in job_titles for role in mid_roles):
                return estimated_ranges['Mid Level']
            else:
                return estimated_ranges['Entry Level']
        
        # Create expected salary analysis
        expected_salary_data = []
        for skill in skill_counts.index:
            skill_jobs = non_numeric_df[non_numeric_df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
            job_count = len(skill_jobs)
            
            if job_count > 0:
                min_expected, max_expected = get_expected_salary_range(skill)
                avg_expected = (min_expected + max_expected) / 2
                
                expected_salary_data.append({
                    'Skill': skill,
                    'Job Count': job_count,
                    'Expected Min': min_expected,
                    'Expected Max': max_expected,
                    'Expected Average': avg_expected
                })
        
        if expected_salary_data:
            expected_df = pd.DataFrame(expected_salary_data).sort_values('Expected Average', ascending=False).head(10)
            expected_df['Expected Average'] = expected_df['Expected Average'].apply(lambda x: f'MUR {x:,.0f}')
            expected_df['Expected Min'] = expected_df['Expected Min'].apply(lambda x: f'MUR {x:,.0f}')
            expected_df['Expected Max'] = expected_df['Expected Max'].apply(lambda x: f'MUR {x:,.0f}')
            st.dataframe(expected_df[['Skill', 'Job Count', 'Expected Average', 'Expected Min', 'Expected Max']], use_container_width=True)
    
    # Overall insights
    st.markdown("### 💡 Comprehensive Skills Insights")
    
    # Get top insights
    most_demanded_skill = skill_counts.index[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>🔥 Most In-Demand Skill</h4>
            <p><strong>{most_demanded_skill}</strong> appearing in {skill_counts[most_demanded_skill]} job postings</p>
            <p>This skill is highly valued across the Mauritian job market</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if len(numeric_salary_df) > 0:
            # Highest paying skill with actual data
            skill_salary_data = []
            for skill in skill_counts.index:
                skill_jobs = numeric_salary_df[numeric_salary_df['skills'].apply(lambda x: isinstance(x, list) and skill in x)]
                if len(skill_jobs) > 0:
                    avg_salary = skill_jobs['salary_average'].mean()
                    skill_salary_data.append({'Skill': skill, 'Average Salary': avg_salary})
            
            if skill_salary_data:
                highest_paying = pd.DataFrame(skill_salary_data).sort_values('Average Salary', ascending=False).iloc[0]
                st.markdown(f"""
                <div class="insight-box">
                    <h4>💰 Highest Paying Skill</h4>
                    <p><strong>{highest_paying['Skill']}</strong> with average salary of MUR {highest_paying['Average Salary']:,.0f}</p>
                    <p>Based on actual salary data from job postings</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
                <h4>📊 Salary Data</h4>
                <p>No numeric salary data available in current filter</p>
                <p>Expected salaries are estimated based on job titles and experience levels</p>
            </div>
            """, unsafe_allow_html=True)

def create_skills_analysis(df):
    """Create skills analysis section"""
    st.subheader("🔧 Most In-Demand Skills")
    
    # Collect all skills
    all_skills = []
    for skills_list in df['skills']:
        if isinstance(skills_list, list):
            all_skills.extend(skills_list)
    
    if not all_skills:
        st.warning("No skills data available for analysis")
        return
    
    # Count skill frequency
    skill_counts = Counter(all_skills)
    top_skills = skill_counts.most_common(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top skills bar chart
        skills, counts = zip(*top_skills)
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(skills, counts)
        ax.set_title('Top 15 Most Requested Skills')
        ax.set_xlabel('Frequency')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{count}', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Skills word cloud
        if all_skills:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(skill_counts)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Skills Word Cloud')
            st.pyplot(fig)
            plt.close()
    
    # Skills insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**Skills Insights:**")
    st.write(f"• **{len(all_skills)}** total skill mentions across {len(df)} job postings")
    st.write(f"• Most requested skill: **{top_skills[0][0]}** ({top_skills[0][1]} mentions)")
    st.write(f"• Top 5 skills represent **{sum(count for _, count in top_skills[:5]) / len(all_skills) * 100:.1f}%** of all skill mentions")
    st.markdown('</div>', unsafe_allow_html=True)

def create_company_analysis(df):
    """Create company analysis section"""
    st.subheader("🏭 Top Companies by Job Postings")
    
    company_counts = df['company_cleaned'].value_counts().head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top companies bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(company_counts.index, company_counts.values)
        ax.set_title('Top 15 Companies by Job Postings')
        ax.set_xlabel('Number of Job Postings')
        
        # Add value labels
        for bar, count in zip(bars, company_counts.values):
            width = bar.get_width()
            ax.text(width + max(company_counts.values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{count}', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Company analysis table
        company_analysis = pd.DataFrame({
            'Company': company_counts.index,
            'Job Postings': company_counts.values,
            'Market Share %': (company_counts.values / len(df) * 100).round(2)
        })
        st.dataframe(company_analysis, use_container_width=True)
    
    # Company insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**Company Insights:**")
    top_company = company_counts.index[0]
    top_company_share = company_counts.iloc[0] / len(df) * 100
    st.write(f"• **{top_company}** leads with **{company_counts.iloc[0]}** postings ({top_company_share:.1f}% market share)")
    st.write(f"• Top 5 companies account for **{(company_counts.head(5).sum() / len(df) * 100):.1f}%** of all job postings")
    st.markdown('</div>', unsafe_allow_html=True)

def create_interactive_filters(df):
    """Create interactive filters section"""
    st.sidebar.subheader("🔍 Interactive Filters")
    
    # Location filter
    locations = ['All'] + list(df['location_cleaned'].unique())
    selected_location = st.sidebar.selectbox('Filter by Location', locations)
    
    # Job type filter
    job_types = ['All'] + list(df['job_type_cleaned'].unique())
    selected_job_type = st.sidebar.selectbox('Filter by Job Type', job_types)
    
    # Experience filter
    experience_levels = ['All'] + list(df['experience_cleaned'].unique())
    selected_experience = st.sidebar.selectbox('Filter by Experience Level', experience_levels)
    
    # Salary range filter
    salary_data = df[df['salary_average'].notna()]
    if len(salary_data) > 0:
        min_salary = int(salary_data['salary_average'].min())
        max_salary = int(salary_data['salary_average'].max())
        salary_range = st.sidebar.slider('Filter by Salary Range (MUR)', min_salary, max_salary, (min_salary, max_salary))
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['location_cleaned'] == selected_location]
    
    if selected_job_type != 'All':
        filtered_df = filtered_df[filtered_df['job_type_cleaned'] == selected_job_type]
    
    if selected_experience != 'All':
        filtered_df = filtered_df[filtered_df['experience_cleaned'] == selected_experience]
    
    if len(salary_data) > 0:
        filtered_df = filtered_df[
            (filtered_df['salary_average'].notna()) & 
            (filtered_df['salary_average'] >= salary_range[0]) & 
            (filtered_df['salary_average'] <= salary_range[1])
        ]
    
    return filtered_df

def create_no_salary_analysis(df):
    """Create analysis for jobs with no salary information"""
    st.subheader("❓ Jobs Without Salary Information")
    
    # Filter jobs with no salary information
    no_salary_df = df[df['salary_cleaned'].isin(['Not specified', 'Negotiable', 'Not disclosed'])].copy()
    
    if len(no_salary_df) == 0:
        st.info("All jobs have salary information available!")
        return
    
    # Overview metrics for no-salary jobs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jobs No Salary Info", f"{len(no_salary_df)} ({len(no_salary_df)/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        negotiable_count = no_salary_df[no_salary_df['salary_cleaned'] == 'Negotiable'].shape[0]
        st.metric("Negotiable", f"{negotiable_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        not_specified_count = no_salary_df[no_salary_df['salary_cleaned'] == 'Not specified'].shape[0]
        st.metric("Not Specified", f"{not_specified_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        unique_companies = no_salary_df['company_cleaned'].nunique()
        st.metric("Unique Companies", f"{unique_companies}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Salary disclosure breakdown
    st.markdown("### Salary Disclosure Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of salary disclosure types
        disclosure_counts = no_salary_df['salary_cleaned'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(disclosure_counts.values, labels=disclosure_counts.index, autopct='%1.1f%%', 
               colors=colors[:len(disclosure_counts)])
        ax.set_title('Salary Disclosure Types')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Bar chart of disclosure types
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(disclosure_counts.index, disclosure_counts.values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax.set_title('Salary Disclosure Types Count')
        ax.set_ylabel('Number of Jobs')
        ax.set_xlabel('Disclosure Type')
        
        # Add value labels
        for bar, count in zip(bars, disclosure_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(disclosure_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Job type analysis for no-salary jobs
    st.markdown("### Job Type Distribution (No Salary Jobs)")
    col1, col2 = st.columns(2)
    
    with col1:
        job_type_counts = no_salary_df['job_type_cleaned'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(job_type_counts.index, job_type_counts.values, color='lightcoral')
        ax.set_title('Job Types Without Salary Info')
        ax.set_ylabel('Number of Jobs')
        ax.set_xlabel('Job Type')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, job_type_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(job_type_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Compare with overall job type distribution
        overall_job_types = df['job_type_cleaned'].value_counts()
        no_salary_job_types = no_salary_df['job_type_cleaned'].value_counts()
        
        comparison_df = pd.DataFrame({
            'Overall': overall_job_types,
            'No Salary': no_salary_job_types
        }).fillna(0)
        
        # Calculate percentage of no-salary jobs by type
        comparison_df['No Salary %'] = (comparison_df['No Salary'] / comparison_df['Overall'] * 100).round(1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        comparison_df['No Salary %'].plot(kind='bar', color='orange', ax=ax)
        ax.set_title('Percentage of Jobs Without Salary by Job Type')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Job Type')
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, pct in enumerate(comparison_df['No Salary %']):
            ax.text(i, pct + max(comparison_df['No Salary %'])*0.01,
                    f'{pct}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    # Location analysis for no-salary jobs
    st.markdown("### Geographic Distribution (No Salary Jobs)")
    col1, col2 = st.columns(2)
    
    with col1:
        location_counts = no_salary_df['location_cleaned'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(location_counts.index, location_counts.values, color='plum')
        ax.set_title('Top 10 Locations - Jobs Without Salary')
        ax.set_xlabel('Number of Jobs')
        
        # Add value labels
        for bar, count in zip(bars, location_counts.values):
            width = bar.get_width()
            ax.text(width + max(location_counts.values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{count}', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Compare location salary disclosure rates
        has_salary = df['salary_cleaned'] != 'Not specified'
        location_salary_disclosure = pd.crosstab(df['location_cleaned'], has_salary)
        location_salary_disclosure.columns = ['No Salary', 'Has Salary']
        location_salary_disclosure['No Salary %'] = (location_salary_disclosure['No Salary'] / 
                                                     (location_salary_disclosure['Has Salary'] + location_salary_disclosure['No Salary']) * 100).round(1)
        
        top_locations_no_salary = location_salary_disclosure.sort_values('No Salary %', ascending=False).head(10)
        
        if len(top_locations_no_salary) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.barh(top_locations_no_salary.index, top_locations_no_salary['No Salary %'], color='gold')
            ax.set_title('Top 10 Locations - Highest No-Salary Percentage')
            ax.set_xlabel('Percentage of Jobs Without Salary (%)')
            
            # Add percentage labels
            for bar, pct in zip(bars, top_locations_no_salary['No Salary %']):
                width = bar.get_width()
                ax.text(width + max(top_locations_no_salary['No Salary %'])*0.01, 
                        bar.get_y() + bar.get_height()/2.,
                        f'{pct}%', ha='left', va='center')
            
            st.pyplot(fig)
            plt.close()
    
    # Experience level analysis
    st.markdown("### Experience Requirements (No Salary Jobs)")
    col1, col2 = st.columns(2)
    
    with col1:
        exp_counts = no_salary_df['experience_cleaned'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(exp_counts.index, exp_counts.values, color='lightblue')
        ax.set_title('Experience Levels - Jobs Without Salary')
        ax.set_ylabel('Number of Jobs')
        ax.set_xlabel('Experience Level')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, exp_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(exp_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Company analysis - top companies with no salary info
        company_counts = no_salary_df['company_cleaned'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(company_counts.index, company_counts.values, color='lightgreen')
        ax.set_title('Top 10 Companies - Jobs Without Salary')
        ax.set_xlabel('Number of Jobs')
        
        # Add value labels
        for bar, count in zip(bars, company_counts.values):
            width = bar.get_width()
            ax.text(width + max(company_counts.values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{count}', ha='left', va='center')
        
        st.pyplot(fig)
        plt.close()
    
    # Data table for no-salary jobs
    st.markdown("### Jobs Without Salary Information")
    display_df = no_salary_df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                               'experience_cleaned', 'salary_cleaned']].copy()
    display_df.columns = ['Title', 'Company', 'Location', 'Job Type', 'Experience', 'Salary Status']
    
    st.dataframe(display_df.head(100), use_container_width=True)
    
    # Key insights
    st.markdown('<div class="no-salary-highlight">', unsafe_allow_html=True)
    st.write("**Key Insights for Jobs Without Salary Information:**")
    st.write(f"• **{len(no_salary_df)}** jobs ({len(no_salary_df)/len(df)*100:.1f}%) don't disclose salary information")
    st.write(f"• **{negotiable_count}** jobs are marked as negotiable ({negotiable_count/len(no_salary_df)*100:.1f}% of no-salary jobs)")
    
    if len(job_type_counts) > 0:
        top_no_salary_type = job_type_counts.index[0]
        st.write(f"• **{top_no_salary_type}** jobs most commonly don't specify salaries ({job_type_counts.iloc[0]} jobs)")
    
    if len(location_counts) > 0:
        top_no_salary_location = location_counts.index[0]
        st.write(f"• **{top_no_salary_location}** has the most jobs without salary info ({location_counts.iloc[0]} jobs)")
    
    st.write("• This lack of salary transparency may indicate:")
    st.write("  - Negotiation-based compensation structures")
    st.write("  - Company policies against salary disclosure")
    st.write("  - Variable compensation based on experience")
    st.write("  - Potential information asymmetry in the job market")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button for no-salary jobs
    csv_no_salary = no_salary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download No-Salary Jobs Dataset",
        data=csv_no_salary,
        file_name='mauritius_jobs_no_salary.csv',
        mime='text/csv'
    )

def create_no_salary_data_table(df):
    """Create data table section for jobs without salary"""
    st.subheader("📊 Jobs Without Salary Information")
    
    # Show sample of no-salary jobs
    st.write("Sample of jobs without salary information:")
    display_df = df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                     'experience_cleaned', 'salary_cleaned', 'skills']].copy()
    display_df.columns = ['Title', 'Company', 'Location', 'Job Type', 'Experience', 'Salary Status', 'Skills']
    
    # Format skills for better display
    display_df['Skills'] = display_df['Skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    
    st.dataframe(display_df.head(100), use_container_width=True)
    
    # Summary tables for no-salary jobs
    st.markdown("### Summary by Category (No-Salary Jobs)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job type summary for no-salary jobs
        job_type_counts = df['job_type_cleaned'].value_counts()
        st.write("**Job Types (No-Salary Jobs):**")
        job_type_df = pd.DataFrame({
            'Count': job_type_counts,
            'Percentage': (job_type_counts / len(df) * 100).round(1)
        })
        st.dataframe(job_type_df, use_container_width=True)
    
    with col2:
        # Location summary for no-salary jobs
        location_counts = df['location_cleaned'].value_counts().head(10)
        st.write("**Top 10 Locations (No-Salary Jobs):**")
        location_df = pd.DataFrame({
            'Count': location_counts,
            'Percentage': (location_counts / len(df) * 100).round(1)
        })
        st.dataframe(location_df, use_container_width=True)
    
    # Download button for no-salary jobs
    csv_no_salary = df.to_csv(index=False)
    st.download_button(
        label="📥 Download No-Salary Jobs Dataset",
        data=csv_no_salary,
        file_name='mauritius_jobs_no_salary.csv',
        mime='text/csv'
    )

def create_data_table(df):
    """Create data table section"""
    st.subheader("📊 Job Listings Data")
    
    # Show sample of data with enhanced salary information
    st.write("Sample of job listings:")
    display_df = df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                     'experience_cleaned', 'salary_cleaned', 'salary_min', 'salary_max', 'salary_average']].copy()
    
    # Format salary columns for better display
    display_df['salary_min'] = display_df['salary_min'].round(0).fillna('')
    display_df['salary_max'] = display_df['salary_max'].round(0).fillna('')
    display_df['salary_average'] = display_df['salary_average'].round(0).fillna('')
    
    # Create formatted salary range column
    def format_salary_range(row):
        if pd.notna(row['salary_min']) and pd.notna(row['salary_max']):
            return f"MUR {int(row['salary_min']):,} - {int(row['salary_max']):,}"
        elif pd.notna(row['salary_average']):
            return f"MUR {int(row['salary_average']):,}"
        elif row['salary_cleaned'] == 'Negotiable':
            return 'Negotiable'
        else:
            return 'Not specified'
    
    display_df['Salary Range'] = display_df.apply(format_salary_range, axis=1)
    
    # Select and rename columns for display
    display_df = display_df[['title', 'company_cleaned', 'location_cleaned', 'job_type_cleaned', 
                            'experience_cleaned', 'Salary Range']]
    display_df.columns = ['Title', 'Company', 'Location', 'Job Type', 'Experience', 'Salary Range']
    
    st.dataframe(display_df.head(100), use_container_width=True)
    
    # Salary summary table
    st.markdown("### Salary Summary by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by job type summary
        salary_by_type = df[df['salary_average'].notna()].groupby('job_type_cleaned')['salary_average'].agg([
            'count', 'mean', 'median', 'min', 'max'
        ]).round(0)
        salary_by_type.columns = ['Count', 'Average', 'Median', 'Min', 'Max']
        salary_by_type = salary_by_type.sort_values('Average', ascending=False)
        
        if len(salary_by_type) > 0:
            st.write("**Average Salary by Job Type:**")
            st.dataframe(salary_by_type, use_container_width=True)
    
    with col2:
        # Salary by location summary
        salary_by_location = df[df['salary_average'].notna()].groupby('location_cleaned')['salary_average'].agg([
            'count', 'mean', 'median', 'min', 'max'
        ]).round(0)
        salary_by_location.columns = ['Count', 'Average', 'Median', 'Min', 'Max']
        salary_by_location = salary_by_location[salary_by_location['Count'] >= 3]  # Filter for meaningful data
        salary_by_location = salary_by_location.sort_values('Average', ascending=False).head(10)
        
        if len(salary_by_location) > 0:
            st.write("**Average Salary by Location (Top 10):**")
            st.dataframe(salary_by_location, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset as CSV",
        data=csv,
        file_name='mauritius_jobs_analysis.csv',
        mime='text/csv'
    )

def main():
    """Main application"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Page navigation in sidebar
    st.sidebar.title("📊 Job Analysis Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Page:",
        ["💰 Jobs With Numeric Salary", "🤝 Jobs With Negotiable Salary", "❓ Jobs Without Salary"]
    )
    
    # Page navigation
    if page == "💰 Jobs With Numeric Salary":
        # Filter jobs with numeric salary information
        numeric_salary_df = df[df['salary_average'].notna()].copy()
        
        # Main header
        st.markdown('<h1 class="main-header">💰 Jobs With Numeric Salary Information</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        if len(numeric_salary_df) == 0:
            st.warning("No jobs with numeric salary information found in the dataset.")
            return
        
        st.info(f"📊 Analyzing {len(numeric_salary_df)} jobs with numeric salary information out of {len(df)} total jobs ({len(numeric_salary_df)/len(df)*100:.1f}%)")
        
        # Interactive filters for numeric salary jobs
        filtered_numeric_salary_df = create_interactive_filters(numeric_salary_df)
        
        # Show filter results
        if len(filtered_numeric_salary_df) != len(numeric_salary_df):
            st.info(f"🔍 Showing {len(filtered_numeric_salary_df)} jobs out of {len(numeric_salary_df)} numeric salary jobs (filtered)")
        
        # Overview metrics for numeric salary jobs
        create_overview_metrics(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Salary analysis
        create_salary_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # High-paying jobs analysis
        create_high_salary_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Job type analysis
        create_job_type_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Location analysis
        create_location_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Experience analysis
        create_experience_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Skills analysis with expected salary insights
        create_skills_with_salary_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Company analysis
        create_company_analysis(filtered_numeric_salary_df)
        st.markdown("---")
        
        # Data table
        create_data_table(filtered_numeric_salary_df)
        
        # Footer
        st.markdown("---")
        st.markdown("### 📝 About Numeric Salary Analysis")
        st.write("""
        This dashboard provides comprehensive analysis of jobs in Mauritian market that disclose specific salary ranges. 
        The analysis includes salary trends, skill requirements with expected salary ranges, job types, locations, and company insights 
        to support data science dissertation research on compensation patterns and skill valuation in Mauritius.
        """)
    
    elif page == "🤝 Jobs With Negotiable Salary":
        # Filter jobs with negotiable salary
        negotiable_df = df[df['salary_cleaned'] == 'Negotiable'].copy()
        
        # Main header
        st.markdown('<h1 class="main-header">🤝 Jobs With Negotiable Salary</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        if len(negotiable_df) == 0:
            st.warning("No jobs with negotiable salary found in the dataset.")
            return
        
        st.info(f"🤝 Analyzing {len(negotiable_df)} jobs with negotiable salary out of {len(df)} total jobs ({len(negotiable_df)/len(df)*100:.1f}%)")
        
        # Interactive filters for negotiable jobs
        filtered_negotiable_df = create_interactive_filters(negotiable_df)
        
        # Show filter results
        if len(filtered_negotiable_df) != len(negotiable_df):
            st.info(f"🔍 Showing {len(filtered_negotiable_df)} jobs out of {len(negotiable_df)} negotiable jobs (filtered)")
        
        # Overview metrics for negotiable jobs
        create_overview_metrics(filtered_negotiable_df)
        st.markdown("---")
        
        # Job type analysis for negotiable jobs
        create_job_type_analysis(filtered_negotiable_df)
        st.markdown("---")
        
        # Location analysis for negotiable jobs
        create_location_analysis(filtered_negotiable_df)
        st.markdown("---")
        
        # Experience analysis for negotiable jobs
        create_experience_analysis(filtered_negotiable_df)
        st.markdown("---")
        
        # Skills analysis with expected salary ranges for negotiable jobs
        create_skills_with_expected_salary_analysis(filtered_negotiable_df, "Negotiable")
        st.markdown("---")
        
        # Company analysis for negotiable jobs
        create_company_analysis(filtered_negotiable_df)
        st.markdown("---")
        
        # Data table for negotiable jobs
        create_negotiable_data_table(filtered_negotiable_df)
        
        # Footer
        st.markdown("---")
        st.markdown("### 📝 About Negotiable Salary Analysis")
        st.write("""
        This analysis examines jobs with negotiable salaries to understand which roles and skills typically offer flexible compensation. 
        Despite not having fixed salary ranges, these jobs provide valuable insights into skill requirements and market segments 
        that prioritize negotiation-based compensation, often indicating senior roles or specialized positions.
        """)
    
    elif page == "❓ Jobs Without Salary":
        # Filter jobs without salary information (not disclosed + not specified)
        no_salary_df = df[(df['salary_cleaned'] == 'Not disclosed') | (df['salary_cleaned'] == 'Not specified')].copy()
        
        # Main header
        st.markdown('<h1 class="main-header">❓ Jobs Without Salary Information</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        if len(no_salary_df) == 0:
            st.info("All jobs have salary information available!")
            return
        
        st.info(f"❓ Analyzing {len(no_salary_df)} jobs without salary information out of {len(df)} total jobs ({len(no_salary_df)/len(df)*100:.1f}%)")
        
        # Interactive filters for no salary jobs
        filtered_no_salary_df = create_interactive_filters(no_salary_df)
        
        # Show filter results
        if len(filtered_no_salary_df) != len(no_salary_df):
            st.info(f"🔍 Showing {len(filtered_no_salary_df)} jobs out of {len(no_salary_df)} no-salary jobs (filtered)")
        
        # Overview metrics for no salary jobs
        create_overview_metrics(filtered_no_salary_df)
        st.markdown("---")
        
        # Job type analysis for no salary jobs
        create_job_type_analysis(filtered_no_salary_df)
        st.markdown("---")
        
        # Location analysis for no salary jobs
        create_location_analysis(filtered_no_salary_df)
        st.markdown("---")
        
        # Experience analysis for no salary jobs
        create_experience_analysis(filtered_no_salary_df)
        st.markdown("---")
        
        # Skills analysis for no salary jobs
        create_skills_analysis(filtered_no_salary_df)
        st.markdown("---")
        
        # Company analysis for no salary jobs
        create_company_analysis(filtered_no_salary_df)
        st.markdown("---")
        
        # Data table for no salary jobs
        create_data_table(filtered_no_salary_df)
        
        # Footer
        st.markdown("---")
        st.markdown("### 📝 About No Salary Analysis")
        st.write("""
        This analysis examines jobs without salary disclosure to understand patterns of salary transparency 
        in the Mauritian job market. Despite lack of compensation information, these jobs provide valuable 
        insights into skill requirements, job types, locations, and company hiring patterns that may indicate 
        negotiation-based compensation or company policies against salary disclosure.
        """)

if __name__ == "__main__":
    main()
