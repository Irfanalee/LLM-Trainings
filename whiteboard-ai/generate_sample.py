"""
Generate synthetic whiteboard images for testing
Creates realistic-looking meeting notes with handwritten-style text
"""

from PIL import Image, ImageDraw, ImageFont
import random
import os

def generate_sample_whiteboard(
    output_path: str = "sample_whiteboard_0.jpg",
    width: int = 1920,
    height: int = 1080
):
    """
    Generate a realistic whiteboard image with meeting notes
    """
    
    # Create white background (whiteboard)
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a handwriting-style font, fallback to default
    try:
        # This font might need to be installed
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Add some texture (slight noise to simulate real whiteboard)
    # Skipped for simplicity, but could add with numpy
    
    # Title
    draw.text((50, 50), "Q4 Planning Meeting", fill='black', font=title_font)
    draw.text((50, 120), "November 15, 2024", fill='blue', font=body_font)
    
    # Draw a dividing line
    draw.line([(40, 180), (width-40, 180)], fill='gray', width=3)
    
    # Section 1: Agenda
    y_pos = 220
    draw.text((50, y_pos), "Agenda:", fill='red', font=header_font)
    y_pos += 60
    
    agenda_items = [
        "1. Review Q3 performance",
        "2. Q4 goals and targets",
        "3. Resource allocation",
        "4. Action items"
    ]
    
    for item in agenda_items:
        draw.text((80, y_pos), item, fill='black', font=body_font)
        y_pos += 45
    
    # Section 2: Action Items (the main focus)
    y_pos += 40
    draw.text((50, y_pos), "Action Items:", fill='red', font=header_font)
    y_pos += 60
    
    action_items = [
        "TODO: Sarah - finalize Q4 budget by Friday (HIGH PRIORITY)",
        "TODO: Mike - hire 2 engineers by month end",
        "TODO: @Team - review new product prototype next week",
        "Action: Lisa - update marketing strategy by Dec 1",
        "TODO: John - complete security audit by Nov 30"
    ]
    
    for item in action_items:
        # Draw checkbox
        draw.rectangle([(80, y_pos-5), (105, y_pos+20)], outline='blue', width=2)
        draw.text((120, y_pos), item, fill='black', font=body_font)
        y_pos += 55
    
    # Section 3: Notes
    y_pos += 40
    draw.text((50, y_pos), "Notes:", fill='red', font=header_font)
    y_pos += 60
    
    notes = [
        "- Budget approved pending Sarah's review",
        "- Need to expedite hiring process",
        "- Product launch target: Q1 2025",
        "- Weekly check-ins on Fridays at 2pm"
    ]
    
    for note in notes:
        draw.text((80, y_pos), note, fill='darkgray', font=body_font)
        y_pos += 45
    
    # Add some handwritten-style imperfections
    # (In real implementation, could add slight rotations, varying sizes, etc.)
    
    # Save
    image.save(output_path, quality=95)
    print(f"✅ Generated sample whiteboard: {output_path}")
    return output_path


def generate_multiple_samples():
    """Generate several different whiteboard scenarios"""
    
    scenarios = [
        {
            "name": "sample_sprint_planning.jpg",
            "title": "Sprint Planning - Week 47",
            "sections": [
                ("User Stories", [
                    "US-101: Implement login page",
                    "US-102: Add password reset",
                    "US-103: User profile editing",
                    "US-104: Email notifications"
                ]),
                ("Action Items", [
                    "TODO: @Dev team - complete US-101 by Wednesday",
                    "TODO: Alice - design mockups by tomorrow",
                    "TODO: Bob - setup CI/CD pipeline this week",
                    "Action: Team - daily standups at 9am"
                ])
            ]
        },
        {
            "name": "sample_brainstorm.jpg",
            "title": "Product Ideas Brainstorm",
            "sections": [
                ("Ideas", [
                    "• AI-powered analytics dashboard",
                    "• Mobile app redesign",
                    "• Customer feedback portal",
                    "• Integration with Slack"
                ]),
                ("Next Steps", [
                    "TODO: Research team - analyze competitors by Friday",
                    "TODO: @Product - create user survey",
                    "Action: Schedule follow-up meeting next week"
                ])
            ]
        }
    ]
    
    for scenario in scenarios:
        # Similar generation logic with different content
        # Simplified here for brevity
        generate_sample_whiteboard(output_path=scenario["name"])
    
    print(f"\n✅ Generated {len(scenarios)} sample whiteboards")


if __name__ == "__main__":
    print("="*60)
    print("Generating Sample Whiteboard Images")
    print("="*60)
    print()
    
    # Generate main sample
    generate_sample_whiteboard()
    #generate_multiple_samples()
    
    print("\nTo generate more samples, uncomment:")
    print("# generate_multiple_samples()")
    
    print("\n" + "="*60)
    print("Now you can test the AI with:")
    print("  python -c 'from whiteboard_ai import WhiteboardAI; ai = WhiteboardAI(); results = ai.analyze_whiteboard(\"sample_whiteboard_0.jpg\"); print(results)'")
    print("="*60)
