"""AI Content Creator Agent - Autonomous content generation and optimization"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ContentType(Enum):
    """Supported content types"""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    VIDEO_SCRIPT = "video_script"
    PRODUCT_DESCRIPTION = "product_description"


class ContentCreatorAgent:
    """Autonomous agent for AI-powered content creation and optimization"""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        use_openai: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize the Content Creator Agent
        
        Args:
            model: Model to use (Claude or OpenAI)
            use_openai: Whether to use OpenAI instead of Anthropic
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_openai = use_openai
        self.client = None
        self.content_history = []
        
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on configuration"""
        if self.use_openai:
            if OpenAI is None:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
        else:
            if anthropic is None:
                raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)

    def generate_content(
        self,
        topic: str,
        content_type: ContentType,
        audience: str = "general",
        tone: str = "professional",
        additional_instructions: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate content based on topic and specifications
        
        Args:
            topic: Main topic for content generation
            content_type: Type of content to generate
            audience: Target audience
            tone: Tone of the content
            additional_instructions: Additional generation instructions
            
        Returns:
            Dictionary containing generated content and metadata
        """
        prompt = self._build_prompt(
            topic, content_type, audience, tone, additional_instructions
        )
        
        response = self._call_llm(prompt)
        
        result = {
            "topic": topic,
            "content_type": content_type.value,
            "content": response,
            "audience": audience,
            "tone": tone,
            "timestamp": datetime.now().isoformat()
        }
        
        self.content_history.append(result)
        return result

    def _build_prompt(
        self,
        topic: str,
        content_type: ContentType,
        audience: str,
        tone: str,
        additional_instructions: Optional[str]
    ) -> str:
        """Build the prompt for content generation"""
        base_prompt = f"""
You are an expert content creator. Generate {content_type.value} content.

Topic: {topic}
Target Audience: {audience}
Tone: {tone}

Requirements:
- Content should be engaging and well-structured
- Optimize for the specified content type
- Include relevant keywords and CTAs where appropriate
- Maintain consistent tone throughout
"""
        if additional_instructions:
            base_prompt += f"\nAdditional Instructions: {additional_instructions}"
        
        return base_prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to generate content"""
        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.content[0].text

    def optimize_content(
        self,
        content: str,
        optimization_type: str = "seo"
    ) -> Dict[str, str]:
        """
        Optimize existing content
        
        Args:
            content: Content to optimize
            optimization_type: Type of optimization (seo, readability, engagement)
            
        Returns:
            Dictionary with optimized content
        """
        prompt = f"""
Optimize the following content for {optimization_type}:

{content}

Provide improvements that enhance {optimization_type} without losing the original message.
"""
        optimized = self._call_llm(prompt)
        return {
            "original": content,
            "optimized": optimized,
            "optimization_type": optimization_type,
            "timestamp": datetime.now().isoformat()
        }

    def get_content_history(self) -> List[Dict]:
        """Get the history of generated content"""
        return self.content_history


def main():
    """Main function demonstrating agent usage"""
    # Initialize agent with Claude
    agent = ContentCreatorAgent()
    
    # Generate blog post content
    blog_result = agent.generate_content(
        topic="The Future of AI in Content Creation",
        content_type=ContentType.BLOG_POST,
        audience="tech professionals",
        tone="informative",
        additional_instructions="Include real-world examples and actionable insights"
    )
    
    print(f"Generated Blog Post:\n{blog_result['content']}\n")
    
    # Optimize content for SEO
    seo_result = agent.optimize_content(
        content=blog_result['content'],
        optimization_type="seo"
    )
    
    print(f"SEO Optimized Content:\n{seo_result['optimized']}")


if __name__ == "__main__":
    main()
