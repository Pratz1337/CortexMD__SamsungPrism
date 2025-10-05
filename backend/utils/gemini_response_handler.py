"""
Utility for handling Gemini API responses and errors
"""

def validate_gemini_response(response):
    """
    Validates a Gemini API response and provides detailed error information.
    
    Args:
        response: The response object from Gemini API
        
    Returns:
        str: The response text if valid
        
    Raises:
        ValueError: If the response is invalid with detailed error message
    """
    
    # Check if response has candidates
    if not hasattr(response, 'candidates') or not response.candidates:
        error_msg = "Gemini API returned no candidates."
        
        # Check for prompt feedback which indicates why the prompt was blocked
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback = response.prompt_feedback
            
            # Check for block reason
            if hasattr(feedback, 'block_reason') and feedback.block_reason:
                error_msg += f" Block reason: {feedback.block_reason}"
            
            # Check for safety ratings
            if hasattr(feedback, 'safety_ratings') and feedback.safety_ratings:
                safety_issues = []
                for rating in feedback.safety_ratings:
                    if hasattr(rating, 'category') and hasattr(rating, 'probability'):
                        if rating.probability in ['HIGH', 'MEDIUM']:
                            safety_issues.append(f"{rating.category}: {rating.probability}")
                
                if safety_issues:
                    error_msg += f" Safety concerns: {', '.join(safety_issues)}"
        
        error_msg += " Check the `response.prompt_feedback` to see if the prompt was blocked. The prompt may have been blocked by safety filters. This often happens with medical content. Consider using more clinical terminology and avoiding detailed symptom descriptions."
        raise ValueError(error_msg)
    
    # Check if response has parts - parts are nested in candidates
    if not response.candidates:
        raise ValueError("Gemini API response has no candidates. The response may have been filtered.")
    
    candidate = response.candidates[0]
    if not hasattr(candidate, 'content') or not candidate.content:
        raise ValueError("Gemini API response candidate has no content.")
    
    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
        raise ValueError("Gemini API response has no content parts. The response may have been filtered.")
    
    # Try to get the text content
    try:
        return response.text
    except Exception as e:
        # Handle specific error case where parts accessor fails due to no candidates
        if "quick accessor only works for a single candidate, but none were returned" in str(e):
            # Provide more detailed error information
            error_details = "The Gemini API response has no candidates available. This often occurs with multiple medical images due to safety filters."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                error_details += " Check response.prompt_feedback for blocking reasons."
            # Don't raise an exception here - let the calling code handle the retry logic
            raise ValueError(f"Multi-image safety filter triggered: {error_details}")
        
        # If response.text fails, try alternative methods
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, 'text'):
                            return part.text
            
            raise ValueError(f"Unable to extract text from Gemini API response structure: {str(e)}")
        except Exception as inner_e:
            raise ValueError(f"Unable to extract text from Gemini API response: {str(e)}. Alternative extraction also failed: {str(inner_e)}")


def safe_generate_content(model, content, max_retries=2):
    """
    Safely generate content with Gemini API with retry logic.
    
    Args:
        model: The Gemini model instance
        content: The content to send (prompt and optional images)
        max_retries: Maximum number of retries
        
    Returns:
        str: The response text
        
    Raises:
        ValueError: If all retries fail
    """
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            print(f"🔄 Gemini API attempt {attempt + 1}/{max_retries + 1}")
            
            # Debug: Log content being sent (truncated for privacy)
            if isinstance(content, list):
                text_content = [item for item in content if isinstance(item, str)]
                if text_content:
                    first_text = text_content[0][:200] + "..." if len(text_content[0]) > 200 else text_content[0]
                    print(f"📝 Sending text content: {first_text}")
                print(f"📁 Content items: {len(content)} (text + images)")
            else:
                content_preview = content[:200] + "..." if len(str(content)) > 200 else str(content)
                print(f"📝 Sending content: {content_preview}")
            
            response = model.generate_content(content)
            
            # Debug: Check response structure
            print(f"📊 Response structure: candidates={hasattr(response, 'candidates')}")
            if hasattr(response, 'candidates'):
                print(f"    candidates_count={len(response.candidates) if response.candidates else 0}")
            if hasattr(response, 'prompt_feedback'):
                print(f"    has_prompt_feedback={response.prompt_feedback is not None}")
                if response.prompt_feedback:
                    print(f"    prompt_feedback_details={response.prompt_feedback}")
            
            return validate_gemini_response(response)
            
        except Exception as e:
            last_error = e
            print(f"❌ Gemini API attempt {attempt + 1} failed: {str(e)}")
            
            # If it's a safety filter issue with multiple images, provide helpful info but don't retry
            if "safety" in str(e).lower() or "blocked" in str(e).lower() or "candidates" in str(e).lower():
                print("🚫 Safety filter or candidate issue detected - not retrying")
                # Check if this might be a multi-image issue
                if isinstance(content, list) and len([item for item in content if hasattr(item, 'save')]) > 1:
                    print("💡 Multi-image content detected - consider processing images individually")
                break
            
            # If it's the last attempt, don't wait
            if attempt == max_retries:
                break
                
            # Wait a bit before retrying
            import time
            time.sleep(1)
    
    # All retries failed
    raise ValueError(f"Gemini API failed after {max_retries + 1} attempts. Last error: {str(last_error)}")