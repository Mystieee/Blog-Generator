from src.states.blogstate import BlogState
from langchain_core.messages import HumanMessage
from src.states.blogstate import Blog

class BlogNode:
    """
    A class to represent the blog node
    """

    def __init__(self, llm):
        self.llm = llm


    def title_creation(self, state: BlogState):
        """
        create title for the blog
        """
        if "topic" in state and state["topic"]:
            prompt ="""
                    You are an expert blog content writer, use markdown formatting.
                    Generate a blog title for the {topic}.
                    This title should be creative and seo friendly.
                    """

            system_message = prompt.format(topic = state["topic"])
            response = self.llm.invoke(system_message)

            return {"blog":{"title": response.content}}

    def content_generation(self, state: BlogState):
        """
        create title for the blog
        """
        if "topic" in state and state["topic"]:
            prompt ="""
                    You are an expert blog content writer, use markdown formatting.
                    Generate a detailed blog content with the detailed breakdown for the {topic}.
                    """

            system_message = prompt.format(topic = state["topic"])
            response = self.llm.invoke(system_message)

            return {"blog":{"title": state['blog']['title'], "content": response.content}}

    def translation(self, state: BlogState):
        """
        Translate the content to the specified language.
        """
        translation_prompt ="""
        Transalate the following content into {current_language}.
        Maintain original tone, style, and formatting
        Adapt cultural references and idioms to be appropriate for {current_language}.

        ORIGINAL CONTENT:
        {blog_content}
        """

        blog_content = state["blog"]["content"]
        messages = [
            HumanMessage(translation_prompt.format(current_language = state["current_language"], blog_content=blog_content))
        ]

        transation_content = self.llm.with_structured_output(Blog, method="json_mode").invoke(messages)

    def route(self, state: BlogState):
        return {"current_language": state['current_language']}

    def route_decision(self, state:BlogState):
        """
        Route the content to the respective transalation function.
        """

        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french":
            return "french"
        else:
            return state["current_language"]