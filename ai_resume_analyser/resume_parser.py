import re
import io
import datetime
from typing import Dict, List, Any, Optional
import pypdf
import docx

# Comprehensive Skills Dictionary categorized by domain
SKILLS_DATABASE = {
    # Programming Languages
    "python": "Python", "java": "Java", "c++": "C++", "c#": "C#", "c": "C", 
    "javascript": "JavaScript", "typescript": "TypeScript", "php": "PHP", 
    "ruby": "Ruby", "swift": "Swift", "kotlin": "Kotlin", "go": "Go", 
    "golang": "Go", "rust": "Rust", "r": "R", "scala": "Scala", "matlab": "MATLAB",
    "dart": "Dart", "shell": "Shell Scripting", "bash": "Bash", "powershell": "PowerShell",
    
    # Web & Frontend
    "html": "HTML5", "html5": "HTML5", "css": "CSS3", "css3": "CSS3", 
    "react": "React.js", "reactjs": "React.js", "react.js": "React.js", 
    "angular": "Angular", "angularjs": "Angular", "vue": "Vue.js", "vuejs": "Vue.js",
    "next.js": "Next.js", "nextjs": "Next.js", "nuxt": "Nuxt.js", "svelte": "Svelte",
    "tailwind": "Tailwind CSS", "tailwindcss": "Tailwind CSS", "bootstrap": "Bootstrap",
    "sass": "Sass", "redux": "Redux", "jquery": "jQuery", "webpack": "Webpack", "vite": "Vite",
    
    # Backend & Frameworks
    "node.js": "Node.js", "nodejs": "Node.js", "express": "Express.js", "expressjs": "Express.js",
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "spring": "Spring Boot",
    "spring boot": "Spring Boot", "springboot": "Spring Boot", "asp.net": "ASP.NET", 
    ".net": ".NET", "laravel": "Laravel", "rails": "Ruby on Rails", "nest.js": "NestJS", "nestjs": "NestJS",
    "graphql": "GraphQL", "rest": "REST API", "restful": "REST API", "rest api": "REST API",
    "grpc": "gRPC", "microservices": "Microservices",
    
    # AI / ML / Data Science
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence", "ai": "AI",
    "data science": "Data Science", "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "llm": "LLMs", "large language models": "LLMs", "generative ai": "Generative AI", "genai": "Generative AI",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "keras": "Keras",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn",
    "pandas": "Pandas", "numpy": "NumPy", "scipy": "SciPy", "matplotlib": "Matplotlib",
    "seaborn": "Seaborn", "plotly": "Plotly", "huggingface": "Hugging Face",
    "transformers": "Transformers", "langchain": "LangChain", "llamaindex": "LlamaIndex",
    "opencv": "OpenCV", "xgboost": "XGBoost", "lightgbm": "LightGBM", "spacy": "spaCy", "nltk": "NLTK",
    
    # Databases & Big Data
    "sql": "SQL", "mysql": "MySQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mongodb": "MongoDB", "redis": "Redis", "sqlite": "SQLite", "oracle": "Oracle DB",
    "cassandra": "Cassandra", "elasticsearch": "Elasticsearch", "dynamodb": "DynamoDB",
    "neo4j": "Neo4j", "snowflake": "Snowflake", "spark": "Apache Spark", "pyspark": "PySpark",
    "hadoop": "Hadoop", "kafka": "Apache Kafka", "tableau": "Tableau", "power bi": "Power BI", "powerbi": "Power BI",
    
    # Cloud & DevOps
    "aws": "AWS", "amazon web services": "AWS", "azure": "Microsoft Azure", "gcp": "Google Cloud (GCP)",
    "google cloud": "Google Cloud (GCP)", "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "ci/cd": "CI/CD", "jenkins": "Jenkins", "github actions": "GitHub Actions", "gitlab ci": "GitLab CI",
    "terraform": "Terraform", "ansible": "Ansible", "linux": "Linux", "git": "Git", "github": "GitHub", "gitlab": "GitLab",
    
    # Core Engineering & Practices
    "agile": "Agile", "scrum": "Scrum", "jira": "Jira", "system design": "System Design",
    "data structures": "Data Structures & Algorithms", "algorithms": "Algorithms", "dsa": "DSA",
    "unit testing": "Unit Testing", "pytest": "PyTest", "tdd": "TDD", "oop": "Object-Oriented Programming"
}

def extract_text_from_file(uploaded_file) -> str:
    """Extract raw text from uploaded PDF, DOCX, or TXT file."""
    try:
        filename = uploaded_file.name.lower()
        
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
            
        elif filename.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs if p.text])
            
        elif filename.endswith(".txt"):
            content = uploaded_file.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="ignore")
            return str(content)
            
        else:
            # Fallback text decoding
            content = uploaded_file.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="ignore")
            return str(content)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def extract_email(text: str) -> Optional[str]:
    """Extract candidate email using regex."""
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    matches = re.findall(email_regex, text)
    return matches[0] if matches else None

def extract_phone(text: str) -> Optional[str]:
    """Extract phone number using regex patterns."""
    phone_regex = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    matches = re.findall(phone_regex, text)
    return matches[0] if matches else None

def extract_links(text: str) -> Dict[str, Optional[str]]:
    """Extract GitHub, LinkedIn, and Portfolio links."""
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
    
    return {
        "github": github_match.group(0) if github_match else None,
        "github_user": github_match.group(1) if github_match else None,
        "linkedin": linkedin_match.group(0) if linkedin_match else None
    }

def extract_name(text: str) -> str:
    """Extract candidate name from the top header lines of the resume."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:6]:
        # Avoid lines that are headings or contact info
        if "@" in line or "http" in line or "resume" in line.lower() or "curriculum" in line.lower():
            continue
        # Check if line looks like a valid name (2 to 4 capitalized words)
        words = line.split()
        if 2 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
            return line
    return "Candidate"

def extract_skills_from_text(text: str) -> List[str]:
    """Extract matched skills from text based on skills database."""
    text_lower = " " + re.sub(r"[^\w\s\+\#\.\-]", " ", text.lower()) + " "
    found_skills = set()
    
    # Multi-word skills first (e.g. 'machine learning', 'deep learning', 'react.js')
    sorted_skills = sorted(SKILLS_DATABASE.keys(), key=lambda x: len(x), reverse=True)
    
    for skill_key in sorted_skills:
        # Regex word boundary check (handling special chars like c++, c#, .net)
        escaped_key = re.escape(skill_key)
        pattern = r"(?:\b|\s)" + escaped_key + r"(?:\b|\s)"
        if re.search(pattern, text_lower):
            found_skills.add(SKILLS_DATABASE[skill_key])
            
    return sorted(list(found_skills))

def extract_education_level(text: str) -> str:
    """Detect highest education level in text: PhD, Masters, Bachelors, or High School."""
    text_lower = text.lower()
    
    # PhD / Doctorate
    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", text_lower):
        return "PhD"
        
    # Masters
    if re.search(r"\b(master|masters|m\.?s|m\.?tech|m\.?sc|mba|mca|m\.?e)\b", text_lower):
        return "Masters"
        
    # Bachelors
    if re.search(r"\b(bachelor|bachelors|b\.?s|b\.?tech|b\.?sc|b\.?e|bca|bba|undergraduate)\b", text_lower):
        return "Bachelors"
        
    # High School default
    return "High School"

def extract_years_experience(text: str) -> int:
    """Estimate total years of experience from experience section or explicit mentions."""
    text_lower = text.lower()
    
    # Check for explicit statements like '1 year of hands-on experience', '5+ years of experience'
    exp_mentions = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?(?:hands-on\s+|industry\s+|professional\s+|relevant\s+)?experience|\s+exp)", text_lower)
    if exp_mentions:
        values = [int(v) for v in exp_mentions if int(v) <= 40]
        if values:
            return max(values)

    # Isolate Experience / Employment section to avoid picking up high school/college education dates
    exp_section = re.search(r"(?:experience|employment|work history|professional experience)([\s\S]{50,2000})(?:education|projects?|skills|certifications|awards|$)", text_lower)
    search_target = exp_section.group(1) if exp_section else text_lower

    # Check for explicit mention in experience section
    sec_mentions = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", search_target)
    if sec_mentions:
        valid_m = [int(v) for v in sec_mentions if 1 <= int(v) <= 30]
        if valid_m:
            return max(valid_m)

    # Extract year spans like 2021 - Present, 2019 - 2021 inside experience section
    current_year = datetime.datetime.now().year
    year_ranges = re.findall(r"\b(20[0-2]\d|199\d)\s*(?:-|–|to)\s*(20[0-2]\d|present|current)\b", search_target)
    
    total_years = 0
    if year_ranges:
        for start_str, end_str in year_ranges:
            start_yr = int(start_str)
            end_yr = current_year if end_str.lower() in ["present", "current"] else int(end_str)
            if end_yr >= start_yr:
                total_years += min(end_yr - start_yr, 15)
        if total_years > 0:
            return min(total_years, 35)
            
    # Default reasonable estimation based on education/length
    words = len(text.split())
    if words > 500:
        return 3
    elif words > 250:
        return 1
    return 0

def extract_project_count(text: str) -> int:
    """Estimate project count by looking at project sections, bullet points, or repo links."""
    text_lower = text.lower()
    
    # Check if a Project section exists
    project_section = re.search(r"(?:projects?|key projects|personal projects|academic projects)([\s\S]{50,1500})(?:experience|education|skills|certifications|$)", text_lower)
    if project_section:
        sec_text = project_section.group(1)
        # Count bullets or project title lines
        bullets = len(re.findall(r"(?:^|\n)\s*[-•*▪]\s+[A-Z0-9]", sec_text))
        named_projects = len(re.findall(r"(?:project\s*\d+|title:|technologies:)", sec_text))
        count = max(bullets, named_projects, 3)
        return min(max(count, 2), 25)
        
    # Check general bullet count or keyword count
    project_mentions = len(re.findall(r"\bproject\b", text_lower))
    if project_mentions >= 3:
        return min(project_mentions + 2, 20)
        
    return 4  # Typical average

def estimate_github_activity(text: str, links: Dict[str, Optional[str]]) -> int:
    """Estimate a candidate's GitHub / coding activity score (0 - 1000)."""
    score = 150 # Base score for having software engineering resume
    
    if links.get("github"):
        score += 350
        
    text_lower = text.lower()
    code_indicators = ["git", "docker", "ci/cd", "unit test", "open source", "repository", "deployed", "api", "pipeline"]
    for indicator in code_indicators:
        if indicator in text_lower:
            score += 35
            
    return min(max(score, 50), 950)

def parse_resume(uploaded_file) -> Dict[str, Any]:
    """Full parsing pipeline for a single resume file."""
    raw_text = extract_text_from_file(uploaded_file)
    word_count = len(raw_text.split())
    
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    links = extract_links(raw_text)
    skills = extract_skills_from_text(raw_text)
    education = extract_education_level(raw_text)
    experience = extract_years_experience(raw_text)
    projects = extract_project_count(raw_text)
    github_activity = estimate_github_activity(raw_text, links)
    
    return {
        "filename": getattr(uploaded_file, "name", "Uploaded Resume"),
        "raw_text": raw_text,
        "name": name,
        "email": email or "Not detected",
        "phone": phone or "Not detected",
        "links": links,
        "skills": skills,
        "education": education,
        "years_experience": experience,
        "project_count": projects,
        "resume_length": max(word_count, 100),
        "github_activity": github_activity
    }
