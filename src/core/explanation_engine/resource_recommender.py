from dataclasses import dataclass
import urllib.parse

@dataclass
class Resource:
    title: str
    url: str
    type: str  # 'video', 'article', 'exercise'
    description: str
    difficulty: str



class ResourceRecommender:
    """Recommends learning resources based on knowledge gaps"""

    def __init__(self):
        # In a production app, you'd initialize API keys here
        self.base_youtube_url = "https://www.youtube.com/results?search_query="
        self.base_scholar_url = "https://scholar.google.com/scholar?q="

    def recommend_videos(self, concept: str, level: str) -> Resource:
        """Recommend educational videos via search link generation"""
        query = f"{concept} explanation for {level} level"
        encoded_query = urllib.parse.quote(query)
        
        return Resource(
            title=f"Expert Video: {concept} ({level})",
            url=f"{self.base_youtube_url}{encoded_query}",
            type="video",
            description=f"A curated video lesson to help you master {concept}.",
            difficulty=level
        )

    def recommend_articles(self, concept: str, reading_level: str) -> Resource:
        """Recommend articles and papers"""
        query = f"{concept} primary concepts {reading_level}"
        encoded_query = urllib.parse.quote(query)

        return Resource(
            title=f"Read: Understanding {concept}",
            url=f"{self.base_scholar_url}{encoded_query}",
            type="article",
            description=f"Deep dive into {concept} with structured reading material.",
            difficulty=reading_level
        )

    def recommend_exercises(self, concept: str, difficulty: str) -> Resource:
        """Recommend practice exercises"""
        # Example of directing to a known practice platform
        query = f"{concept} practice problems {difficulty}"
        encoded_query = urllib.parse.quote(query)
        
        return Resource(
            title=f"Practice: {concept} Quiz",
            url=f"https://www.google.com/search?q={encoded_query}+interactive+quiz",
            type="exercise",
            description=f"Test your knowledge of {concept} with interactive challenges.",
            difficulty=difficulty
        )