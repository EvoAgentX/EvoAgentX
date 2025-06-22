PDF_AGENT_SUGGESTION = """
    It would be better to include a short summary on the resume at the beginning of the output.
    You should use the tools to find real job opportunities and read resume.
    I already give you the path in this goal. You should take "goal" as the only input at beginning.
    The output should be well-formatted.
    You should return every detailed information you collected from the website, inclduing job title, company name, location, detailed job description, detailed job requirements, salary, job posting link, etc.
    """

def formulate_goal(goal):
    return f"""
    {goal}
    {PDF_AGENT_SUGGESTION}
    """