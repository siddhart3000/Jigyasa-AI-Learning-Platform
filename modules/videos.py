from __future__ import annotations


def learning_videos() -> dict[str, list[dict[str, str]]]:
    # video_id is used for watch tracking
    return {
        "Maths": [
            {"id": "math_add", "title": "Basic Addition for Kids", "url": "https://www.youtube.com/watch?v=pxUY29LpMZE"},
            {"id": "math_sub", "title": "Subtraction for Kids", "url": "https://www.youtube.com/watch?v=ug0gs8kLE48"},
            {"id": "math_mul", "title": "Multiplication for Beginners", "url": "https://www.youtube.com/watch?v=TKAhQ-9BQnw"},
            {"id": "math_div", "title": "Division for Kids", "url": "https://www.youtube.com/watch?v=ek1JJVYaXxU"},
        ],
        "Science": [
            {"id": "sci_body", "title": "Human Body for Kids", "url": "https://www.youtube.com/watch?v=B3Fv2X8EKfE"},
            {"id": "sci_water", "title": "Water Cycle Explained", "url": "https://www.youtube.com/watch?v=46NNUXgP55k"},
            {"id": "sci_matter", "title": "States of Matter", "url": "https://www.youtube.com/watch?v=wYe6zPKHDWo"},
            {"id": "sci_solar", "title": "Solar System Explained", "url": "https://www.youtube.com/watch?v=mxzyANgHrS0"},
        ],
        "English": [
            {"id": "eng_alpha", "title": "English Alphabet Learning", "url": "https://www.youtube.com/watch?v=k-1dMgvAJMU"},
            {"id": "eng_phonics", "title": "Phonics for Beginners", "url": "https://www.youtube.com/watch?v=CppN8rvb4HA"},
            {"id": "eng_grammar", "title": "Basic Grammar for Kids", "url": "https://www.youtube.com/watch?v=4ncLB3JPy_w"},
        ],
        "General Knowledge": [
            {"id": "gk_facts", "title": "Amazing Facts for Kids", "url": "https://www.youtube.com/watch?v=oZgqf-F4J40"},
            {"id": "gk_flags", "title": "Countries and Flags of the World", "url": "https://www.youtube.com/watch?v=K9MPeW6BBwg"},
            {"id": "gk_capitals", "title": "World Capitals for Kids", "url": "https://www.youtube.com/watch?v=a57VPS8bLKo"},
        ],
        "Fun Activities": [
            {"id": "fun_draw", "title": "Easy Drawing for Kids", "url": "https://www.youtube.com/watch?v=MfidW9ma7zI"},
            {"id": "fun_yoga", "title": "Kids Yoga Session", "url": "https://www.youtube.com/watch?v=X655B4ISakg"},
            {"id": "fun_origami", "title": "Fun Origami for Kids", "url": "https://www.youtube.com/watch?v=SHfhiaezqdg"},
        ],
    }

