import os
import io
import base64
import logging
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
import anthropic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global PDF reader
try:
    import pypdf  # type: ignore
    PDF_READER_CLASS = pypdf.PdfReader
    logger.info("Successfully imported pypdf")
except ImportError:
    logger.warning("Failed to import pypdf, trying PyPDF2")
    try:
        import PyPDF2  # type: ignore
        PDF_READER_CLASS = PyPDF2.PdfReader
        logger.info("Successfully imported PyPDF2")
    except ImportError as e:
        logger.error(f"Failed to import PDF libraries: {e}")
        st.error("PDF processing libraries not available. Please contact support.")
        PDF_READER_CLASS = None

# Load environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

def clean_extracted_text(text: str) -> str:
    """Clean and format extracted PDF text."""
    if not text:
        return ""
    text = text.replace('\n\n', '\n')
    text = text.strip()
    return text

def extract_pdf_content(pdf_path: str) -> Optional[str]:
    """Extract and process text content from a PDF file."""
    if not PDF_READER_CLASS:
        st.error("PDF processing is not available")
        return None

    logger.info(f"Starting PDF extraction from: {pdf_path}")
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        with open(pdf_path, 'rb') as file:
            reader = PDF_READER_CLASS(file)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return clean_extracted_text(" ".join(text))
    except Exception as e:
        logger.error(f"Error in PDF extraction: {str(e)}")
        return None

def get_reference_file(grade: str) -> Optional[str]:
    """
    Map grade levels to their reference PDF filenames.
    These files live in reference_materials/ and have the exact titles:
      (TCAP) science assessments Grade 6 - Google Docs.pdf
      (TCAP) science assessments Grade 7 - Google Docs.pdf
      (TCAP) science assessments Grade 8 - Google Docs.pdf
    """
    grade_mapping = {
        "Grade 6": "(TCAP) science assessments Grade 6 - Google Docs.pdf",
        "Grade 7": "(TCAP) science assessments Grade 7 - Google Docs.pdf",
        "Grade 8": "(TCAP) science assessments Grade 8 - Google Docs.pdf",
        # other grades map to None
        "Kindergarten": None,
        "Grade 1": None,
        "Grade 2": None,
        "Grade 3": None,
        "Grade 4": None,
        "Grade 5": None,
        "Biology": None,
        "Chemistry": None,
        "Physics": None,
    }
    return grade_mapping.get(grade)

def format_response(text: str) -> str:
    """Format the response with custom styling."""
    questions = text.split('Question')[1:]
    formatted_questions = []
    for q in questions:
        formatted_q = f'''
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #1f77b4;
        ">
            Question{q.replace(chr(10), '<br>')}
        </div>
        '''
        formatted_questions.append(formatted_q)
    return "".join(formatted_questions)

def get_response(
    grade: str,
    standards: str,
    figure_out: str,
    overview: str,
    start: int = None,
    end: int = None
) -> str:
    """Generate assessment content using the AI model."""
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load reference material (used internally in the prompt only)
    reference_file = get_reference_file(grade)
    reference_text = ""
    if reference_file:
        pdf_path = f"reference_materials/{reference_file}"
        reference_text = extract_pdf_content(pdf_path) or ""
    
    # build the slice instruction if provided
    slice_instruction = ""
    if start is not None and end is not None:
        slice_instruction = f"\nImportant: Output ONLY Questions {start} through {end} in full, with their solutions. Do NOT ask to continue, do NOT summarize, do NOT output any message except the questions and solutions. Do NOT say 'continue', 'remaining', or anything similar."

    user_content = f"""
# CONTEXT #
You are creating a set of science assessment items for {grade}.

# START IMMEDIATELY WITH QUESTION {start if start else 1} #
# Content Hierarchy (in order of priority):
1. Standards
2. What students will figure out
3. Unit Overview

# Preliminary Steps:
1. Review the Standards, What students will figure out, and Unit Overview.
2. Write a summary of how the Standards and Unit Overview determine what students will figure out.
3. From the Standards and What students will figure out, write a list of skills needed.
   - Show how each skill helps learners master the standard.

# Question Creation Guidelines:
- Generate exactly 53 questions:
  - 5 Multiple Choice (MC) questions
  - 5 Multiple Select (MS) questions
  - 3 of each type of Technology Enhanced (TE) questions for a total of 12 questions
  - 3 Cluster Items questions for a total of 24 questions
  - 3 Evidence-Based Selected Response (EBSR) questions for a total of 15 questions
  - 2 Short Constructed-Response (CR) questions
- Ensure each question is solvable with the information provided.
- Each question must reflect: What students will figure out, then align with the Unit Overview, and finally comply with the Standards.

## Multiple Choice (MC):
Students select one correct answer from four options.
Provide four answer options labeled A, B, C, D.
Each option should appear on its own line.
Each MC item is worth one point.

## Multiple Select (MS):
Students select multiple correct answers from a list of options.
The number of correct answers may or may not be specified.
Provide 5 answer options labeled A, B, C, D, E.
Each option should appear on its own line.
MS items are worth two points, with the possibility of earning partial credit.

## Technology Enhanced (TE):
These items utilize the online testing platform's capabilities, such as drag-and-drop, graphing, hot-spot, inline-choice, or interactive simulations.  
- Drag-and-Drop: Student drags labels onto a diagram. Include a description of the diagram for manual reproduction.  
- Hot-Spot: Student clicks on the correct area of an image. Include a description for manual reproduction.  
- Inline-Choice: Student picks from a drop-down menu embedded in a sentence.  
- Graphing: Student plots points or draws a line on an on-screen grid.  
TE items are designed to assess students' application of scientific concepts in interactive formats.

## Cluster Items:
A set of up to eight related questions based on a common stimulus or scenario.
Clusters assess multiple dimensions of science understanding, including disciplinary core ideas, science and engineering practices, and crosscutting concepts.
Questions in a cluster can be Multiple Choice, Multiple Select, or Graphing.
Each question within a cluster is scored independently.

## Evidence-Based Selected Response Questions
Students answer a multiple-choice question and then select the evidence that best supports that answer.
Generate a phenomenon with Multiple Choice Questions format.

## Constructed-Response (CR) questions
Students write a free-response answer.
Assessed with rubrics.
Require a clear step-by-step solution leading to a concise final answer.

## Visual Descriptions (if needed)
Place all visual descriptions in square brackets: [Visual: ...].
Include enough detail (coordinates, measures, etc.) so the problem is solvable.
Verify geometric or diagram-based details for mathematical consistency.

# Formatting Requirements:
1. Number each question as "Question X: <standard>, <question_type>". Example: "Question 1: 8.ESS2.1, MC"
2. Identify which standard best aligns with the question and add it after the number (e.g., "Question 1: 8.ESS2.1, MC").
3. Use the following codes for question types:
   - Multiple Choice: MC
   - Multiple Select: MS
   - Technology Enhanced: TE - drag-and-drop, TE - hot-spot, TE - inline-choice, TE - graphing
   - Cluster Items: Cluster
   - Evidence-Based Selected Response: EBSR
   - Constructed Response: CR
4. Use the following answer format for all items:
   Answer: [Letter(s) or numeric answer] | Model Solution:
   • rationale  
   • Final answer statement and grading rubric  
5. Do not include any meta-commentary or extra text beyond the questions and their solutions.

# Validation Checklist:
1. Is the question solvable with the provided information?
2. Does it reflect What students will figure out, then align with the Unit Overview, then the Standards?
3. Are visual or geometric details valid and clearly labeled?
4. Is there no missing or extraneous information?

# REFERENCE FORMAT #
The official {grade} assessment items live in these PDFs in our repo’s  
`reference_materials/` folder. Point back to them if you need to quote exact  
wording or structure:

- reference_materials/(TCAP) science assessments Grade 6 - Google Docs.pdf  
- reference_materials/(TCAP) science assessments Grade 7 - Google Docs.pdf  
- reference_materials/(TCAP) science assessments Grade 8 - Google Docs.pdf  

# CONTENT TO ADDRESS #
Generate questions covering:
***Standards: {standards}
***What students will figure out: {figure_out}
***Unit Overview: {overview}
{slice_instruction}
"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=(
            "You are a science assessment writer who exactly replicates official state assessment style and format. "
            "Output ONLY the questions and their solutions. "
            "Do NOT provide summaries, intros, or confirmation prompts."
        ),
        messages=[{"role": "user", "content": user_content}],
        max_tokens=4000,
        stream=False
    )
    return response.content[0].text

# Streamlit UI
st.title("Science Assessment Generator")
st.subheader("Generate structured assessment items with rationales")
st.markdown("""
For samples and more information on how to collect the data for this tool, 
check out this [Help Page](https://docs.google.com/spreadsheets/d/1lXPIsCrwuEH3yAaj5xR56-FloC-pXU03L7gdh4Qe7bg/edit?usp=sharing).
""")

# Grade Level Selection
grade = st.selectbox(
    "Grade Level:", 
    [f"Grade {i}" for i in range(6, 9)] + ["Biology", "Chemistry", "Physics", "Kindergarten"] + [f"Grade {i}" for i in range(1, 6)]
)

# Two-column layout for standards and what students will figure out
col1, col2 = st.columns(2)
with col1:
    standards = st.text_area(
        "Standards:", 
        help="List the relevant content standards being addressed.",
        height=150
    )
with col2:
    figure_out = st.text_area(
        "What Students Will Figure Out:", 
        help="List what students will figure out during this unit.",
        height=150
    )

# Unit Overview
unit_overview = st.text_area(
    "Unit Overview:", 
    help="Provide an overview of the content being covered in this unit.",
    height=200
)

from docx import Document
import io

# Generate response on button click
if st.button("Generate Assessment"):
    if all([grade, standards, figure_out, unit_overview]):
        with st.spinner("Generating assessment in batches of 10 questions..."):
            try:
                import re
                full_text = ""
                batch_size = 10
                total_questions = 53
                current = 1
                while current <= total_questions:
                    end = min(current + batch_size - 1, total_questions)
                    part = get_response(
                        grade,
                        standards,
                        figure_out,
                        unit_overview,
                        start=current,
                        end=end
                    )
                    # Remove unwanted continuation messages
                    part = re.sub(r"\[.*?continu(ing|e|ation).*?\]", "", part, flags=re.IGNORECASE)
                    part = re.sub(r"\[.*?remaining.*?\]", "", part, flags=re.IGNORECASE)
                    part = re.sub(r"\[.*?follow(ing|s).*?\]", "", part, flags=re.IGNORECASE)
                    full_text += part.strip() + "\n\n"
                    # Find the last question number generated
                    matches = list(re.finditer(r"Question (\d+):", part))
                    if matches:
                        last_q = int(matches[-1].group(1))
                        current = last_q + 1
                    else:
                        current = end + 1
                st.success("Assessment Generated Successfully!")
                st.markdown(format_response(full_text), unsafe_allow_html=True)
                with st.expander("Show Raw Text"):
                    st.text_area("Raw Assessment Text", value=full_text, height=400)
                # build and download .docx
                doc = Document()
                for line in full_text.split("\n"):
                    doc.add_paragraph(line)
                buffer = io.BytesIO()
                doc.save(buffer)
                buffer.seek(0)
                st.download_button(
                    label="Download as Word (.docx)",
                    data=buffer,
                    file_name="Science_Assessment.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error generating assessment: {e}")
    else:
        st.warning("Please fill in all fields to generate the assessment.")
