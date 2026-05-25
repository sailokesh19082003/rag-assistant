import os
import logging
from typing import Optional, Tuple
import google.generativeai as genai

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

def generate_response(prompt: str, system_prompt: str) -> Tuple[str, Optional[int]]:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1024}
        )
        reply = response.text
        logger.info("Gemini response received")
        return reply, None
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise RuntimeError(f"Gemini API error: {str(e)}")