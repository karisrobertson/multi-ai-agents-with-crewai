You can download the `requirements.txt` for this course from the workspace of this lab. `File --> Open...`

# L2: Create Agents to Research and Write an Article

In this lesson, you will be introduced to the foundational concepts of multi-agent systems and get an overview of the crewAI framework.

The libraries are already installed in the classroom. If you're running this notebook on your own machine, you can install the following:
```Python
!pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
```


```python
# Warning control
import warnings
warnings.filterwarnings('ignore')
```

- Import from the crewAI libray.


```python
from crewai import Agent, Task, Crew
```

- As a LLM for your agents, you'll be using OpenAI's `gpt-3.5-turbo`.

**Optional Note:** crewAI also allow other popular models to be used as a LLM for your Agents. You can see some of the examples at the [bottom of the notebook](#1).


```python
import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
```

## Creating Agents

- Define your Agents, and provide them a `role`, `goal` and `backstory`.
- It has been seen that LLMs perform better when they are role playing.

### Agent: Planner

**Note**: The benefit of using _multiple strings_ :
```Python
varname = "line 1 of text"
          "line 2 of text"
```

versus the _triple quote docstring_:
```Python
varname = """line 1 of text
             line 2 of text
          """
```
is that it can avoid adding those whitespaces and newline characters, making it better formatted to be passed to the LLM.


```python
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True
)
```

### Agent: Writer


```python
writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)
```

### Agent: Editor


```python
editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True
)
```

## Creating Tasks

- Define your Tasks, and provide them a `description`, `expected_output` and `agent`.

### Task: Plan


```python
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)
```

### Task: Write


```python
write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)
```

### Task: Edit


```python
edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)
```

## Creating the Crew

- Create your crew of Agents
- Pass the tasks to be performed by those agents.
    - **Note**: *For this simple example*, the tasks will be performed sequentially (i.e they are dependent on each other), so the _order_ of the task in the list _matters_.
- `verbose=2` allows you to see all the logs of the execution. 


```python
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)
```

## Running the Crew

**Note**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.


```python
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
```

    [1m[95m [DEBUG]: == Working Agent: Content Planner[00m
    [1m[95m [INFO]: == Starting Task: 1. Prioritize the latest trends, key players, and noteworthy news on Artificial Intelligence.
    2. Identify the target audience, considering their interests and pain points.
    3. Develop a detailed content outline including an introduction, key points, and a call to action.
    4. Include SEO keywords and relevant data or sources.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer: 
    
    Content Plan: 
    
    Title: Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Introduction:
    - Brief overview of Artificial Intelligence (AI) and its impact on various industries
    - Mention of the rapid advancements and innovations in AI technology
    
    Key Points:
    1. Latest Trends in Artificial Intelligence:
    - Deep learning and neural networks
    - Natural language processing
    - AI-based automation and robotics
    - AI in healthcare and medicine
    - Ethical considerations in AI development
    
    2. Key Players in the AI Industry:
    - Google AI
    - IBM Watson
    - Amazon AI
    - Microsoft AI
    - OpenAI
    
    3. Noteworthy News in Artificial Intelligence:
    - Recent breakthroughs in AI research
    - Industry partnerships and collaborations
    - AI applications in solving real-world problems
    - Regulatory updates and policies related to AI
    
    Audience Analysis:
    - Target Audience: Tech enthusiasts, business professionals, students, and anyone interested in emerging technologies
    - Interests: Innovations in AI, industry applications, ethical implications, future prospects
    - Pain Points: Understanding complex AI concepts, staying updated on the latest trends, navigating the AI industry landscape
    
    SEO Keywords:
    - Artificial Intelligence trends 2021
    - Top AI companies
    - AI news and updates
    - Ethical AI development
    - Future of Artificial Intelligence
    
    Call to Action:
    - Encourage readers to subscribe for regular updates on AI trends and news
    - Promote a related webinar, workshop, or online course on AI technology
    - Invite readers to share their thoughts and experiences with AI in the comments section
    
    Resources:
    1. Forbes - "The Latest Trends Shaping Artificial Intelligence in 2021"
    2. TechCrunch - "Top Players in the AI Industry You Need to Know"
    3. Harvard Business Review - "Ethical Considerations in AI Development"
    4. AI Trends Magazine - "Industry News and Updates on Artificial Intelligence"
    
    By following this comprehensive content plan, we aim to provide valuable insights and information on the latest trends, key players, and noteworthy news in the field of Artificial Intelligence.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Planner] Task output: Content Plan: 
    
    Title: Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Introduction:
    - Brief overview of Artificial Intelligence (AI) and its impact on various industries
    - Mention of the rapid advancements and innovations in AI technology
    
    Key Points:
    1. Latest Trends in Artificial Intelligence:
    - Deep learning and neural networks
    - Natural language processing
    - AI-based automation and robotics
    - AI in healthcare and medicine
    - Ethical considerations in AI development
    
    2. Key Players in the AI Industry:
    - Google AI
    - IBM Watson
    - Amazon AI
    - Microsoft AI
    - OpenAI
    
    3. Noteworthy News in Artificial Intelligence:
    - Recent breakthroughs in AI research
    - Industry partnerships and collaborations
    - AI applications in solving real-world problems
    - Regulatory updates and policies related to AI
    
    Audience Analysis:
    - Target Audience: Tech enthusiasts, business professionals, students, and anyone interested in emerging technologies
    - Interests: Innovations in AI, industry applications, ethical implications, future prospects
    - Pain Points: Understanding complex AI concepts, staying updated on the latest trends, navigating the AI industry landscape
    
    SEO Keywords:
    - Artificial Intelligence trends 2021
    - Top AI companies
    - AI news and updates
    - Ethical AI development
    - Future of Artificial Intelligence
    
    Call to Action:
    - Encourage readers to subscribe for regular updates on AI trends and news
    - Promote a related webinar, workshop, or online course on AI technology
    - Invite readers to share their thoughts and experiences with AI in the comments section
    
    Resources:
    1. Forbes - "The Latest Trends Shaping Artificial Intelligence in 2021"
    2. TechCrunch - "Top Players in the AI Industry You Need to Know"
    3. Harvard Business Review - "Ethical Considerations in AI Development"
    4. AI Trends Magazine - "Industry News and Updates on Artificial Intelligence"
    
    By following this comprehensive content plan, we aim to provide valuable insights and information on the latest trends, key players, and noteworthy news in the field of Artificial Intelligence.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Content Writer[00m
    [1m[95m [INFO]: == Starting Task: 1. Use the content plan to craft a compelling blog post on Artificial Intelligence.
    2. Incorporate SEO keywords naturally.
    3. Sections/Subtitles are properly named in an engaging manner.
    4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
    5. Proofread for grammatical errors and alignment with the brand's voice.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI have a clear understanding of the content plan and the key points to cover in the blog post about Artificial Intelligence. I am ready to craft a compelling piece that engages the target audience and provides valuable insights on the latest trends and key players in the AI industry.
    
    Final Answer:
    
    # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Artificial Intelligence (AI) has revolutionized various industries with its capabilities to automate processes, analyze data, and make predictions. The rapid advancements in AI technology have paved the way for innovative solutions that enhance efficiency and drive growth. As we delve into the latest trends and key players in the AI industry, it's essential to understand the impact of AI on society and the ethical considerations that come with its development.
    
    ## Latest Trends in Artificial Intelligence
    
    ### Deep learning and neural networks
    Deep learning algorithms and neural networks have significantly improved AI's ability to learn from data, recognize patterns, and make decisions. These technologies have been instrumental in powering applications like image recognition, speech recognition, and autonomous vehicles.
    
    ### Natural language processing
    Natural language processing (NLP) enables machines to understand and interpret human language, facilitating communication between humans and computers. NLP applications include chatbots, language translation, sentiment analysis, and voice assistants like Siri and Alexa.
    
    ### AI-based automation and robotics
    AI-driven automation and robotics have transformed industries by streamlining processes, increasing productivity, and reducing human error. From manufacturing and logistics to healthcare and finance, AI-powered robots are reshaping the way work is done.
    
    ### AI in healthcare and medicine
    The healthcare industry has embraced AI to improve patient care, diagnosis accuracy, and treatment outcomes. AI applications in healthcare include medical imaging analysis, drug discovery, personalized medicine, and predictive analytics for disease prevention.
    
    ### Ethical considerations in AI development
    As AI technology continues to evolve, it's crucial to address ethical considerations related to data privacy, bias in AI algorithms, transparency in decision-making, and the impact of AI on jobs and society. Ethical AI development frameworks and guidelines are essential to ensure that AI benefits humanity responsibly.
    
    ## Key Players in the AI Industry
    
    ### Google AI
    Google's AI research division, Google AI, is at the forefront of developing cutting-edge AI technologies. From self-driving cars to natural language understanding, Google AI is driving innovation across various domains.
    
    ### IBM Watson
    IBM Watson is a cognitive computing platform that leverages AI to analyze vast amounts of data, extract insights, and make informed decisions. Watson's applications span healthcare, finance, retail, and cybersecurity.
    
    ### Amazon AI
    Amazon's AI initiatives focus on enhancing customer experiences through personalized recommendations, voice recognition, and supply chain optimization. Amazon AI powers Alexa, Amazon Go stores, and Amazon Web Services (AWS) AI services.
    
    ### Microsoft AI
    Microsoft's AI division is dedicated to democratizing AI and making it accessible to businesses and developers worldwide. Microsoft AI solutions include Azure AI, cognitive services, and AI-powered productivity tools.
    
    ### OpenAI
    OpenAI is a research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. OpenAI conducts cutting-edge research in AI safety, ethics, and the development of AGI systems.
    
    ## Noteworthy News in Artificial Intelligence
    
    Recent breakthroughs in AI research have showcased the potential of AI to solve complex problems and drive innovation. Industry partnerships and collaborations are accelerating the adoption of AI technologies across sectors, while regulatory updates and policies are shaping the future of AI development.
    
    In conclusion, Artificial Intelligence continues to shape the future of technology and society, with the latest trends and key players driving innovation and progress. As we navigate the evolving landscape of AI, it's essential to stay informed, engage in ethical discussions, and embrace the transformative power of AI for the betterment of humanity.
    
    *Subscribe to our newsletter for regular updates on AI trends and news. Join our upcoming webinar on Ethical AI Development or share your thoughts and experiences with AI in the comments section below.*[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Writer] Task output: # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Artificial Intelligence (AI) has revolutionized various industries with its capabilities to automate processes, analyze data, and make predictions. The rapid advancements in AI technology have paved the way for innovative solutions that enhance efficiency and drive growth. As we delve into the latest trends and key players in the AI industry, it's essential to understand the impact of AI on society and the ethical considerations that come with its development.
    
    ## Latest Trends in Artificial Intelligence
    
    ### Deep learning and neural networks
    Deep learning algorithms and neural networks have significantly improved AI's ability to learn from data, recognize patterns, and make decisions. These technologies have been instrumental in powering applications like image recognition, speech recognition, and autonomous vehicles.
    
    ### Natural language processing
    Natural language processing (NLP) enables machines to understand and interpret human language, facilitating communication between humans and computers. NLP applications include chatbots, language translation, sentiment analysis, and voice assistants like Siri and Alexa.
    
    ### AI-based automation and robotics
    AI-driven automation and robotics have transformed industries by streamlining processes, increasing productivity, and reducing human error. From manufacturing and logistics to healthcare and finance, AI-powered robots are reshaping the way work is done.
    
    ### AI in healthcare and medicine
    The healthcare industry has embraced AI to improve patient care, diagnosis accuracy, and treatment outcomes. AI applications in healthcare include medical imaging analysis, drug discovery, personalized medicine, and predictive analytics for disease prevention.
    
    ### Ethical considerations in AI development
    As AI technology continues to evolve, it's crucial to address ethical considerations related to data privacy, bias in AI algorithms, transparency in decision-making, and the impact of AI on jobs and society. Ethical AI development frameworks and guidelines are essential to ensure that AI benefits humanity responsibly.
    
    ## Key Players in the AI Industry
    
    ### Google AI
    Google's AI research division, Google AI, is at the forefront of developing cutting-edge AI technologies. From self-driving cars to natural language understanding, Google AI is driving innovation across various domains.
    
    ### IBM Watson
    IBM Watson is a cognitive computing platform that leverages AI to analyze vast amounts of data, extract insights, and make informed decisions. Watson's applications span healthcare, finance, retail, and cybersecurity.
    
    ### Amazon AI
    Amazon's AI initiatives focus on enhancing customer experiences through personalized recommendations, voice recognition, and supply chain optimization. Amazon AI powers Alexa, Amazon Go stores, and Amazon Web Services (AWS) AI services.
    
    ### Microsoft AI
    Microsoft's AI division is dedicated to democratizing AI and making it accessible to businesses and developers worldwide. Microsoft AI solutions include Azure AI, cognitive services, and AI-powered productivity tools.
    
    ### OpenAI
    OpenAI is a research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. OpenAI conducts cutting-edge research in AI safety, ethics, and the development of AGI systems.
    
    ## Noteworthy News in Artificial Intelligence
    
    Recent breakthroughs in AI research have showcased the potential of AI to solve complex problems and drive innovation. Industry partnerships and collaborations are accelerating the adoption of AI technologies across sectors, while regulatory updates and policies are shaping the future of AI development.
    
    In conclusion, Artificial Intelligence continues to shape the future of technology and society, with the latest trends and key players driving innovation and progress. As we navigate the evolving landscape of AI, it's essential to stay informed, engage in ethical discussions, and embrace the transformative power of AI for the betterment of humanity.
    
    *Subscribe to our newsletter for regular updates on AI trends and news. Join our upcoming webinar on Ethical AI Development or share your thoughts and experiences with AI in the comments section below.*
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Editor[00m
    [1m[95m [INFO]: == Starting Task: Proofread the given blog post for grammatical errors and alignment with the brand's voice.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m


    [32;1m[1;3mI now can give a great answer.
    
    Final Answer:
    
    # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Artificial Intelligence (AI) has revolutionized various industries with its capabilities to automate processes, analyze data, and make predictions. The rapid advancements in AI technology have paved the way for innovative solutions that enhance efficiency and drive growth. As we delve into the latest trends and key players in the AI industry, it's essential to understand the impact of AI on society and the ethical considerations that come with its development.
    
    ## Latest Trends in Artificial Intelligence
    
    ### Deep learning and neural networks
    Deep learning algorithms and neural networks have significantly improved AI's ability to learn from data, recognize patterns, and make decisions. These technologies have been instrumental in powering applications like image recognition, speech recognition, and autonomous vehicles.
    
    ### Natural language processing
    Natural language processing (NLP) enables machines to understand and interpret human language, facilitating communication between humans and computers. NLP applications include chatbots, language translation, sentiment analysis, and voice assistants like Siri and Alexa.
    
    ### AI-based automation and robotics
    AI-driven automation and robotics have transformed industries by streamlining processes, increasing productivity, and reducing human error. From manufacturing and logistics to healthcare and finance, AI-powered robots are reshaping the way work is done.
    
    ### AI in healthcare and medicine
    The healthcare industry has embraced AI to improve patient care, diagnosis accuracy, and treatment outcomes. AI applications in healthcare include medical imaging analysis, drug discovery, personalized medicine, and predictive analytics for disease prevention.
    
    ### Ethical considerations in AI development
    As AI technology continues to evolve, it's crucial to address ethical considerations related to data privacy, bias in AI algorithms, transparency in decision-making, and the impact of AI on jobs and society. Ethical AI development frameworks and guidelines are essential to ensure that AI benefits humanity responsibly.
    
    ## Key Players in the AI Industry
    
    ### Google AI
    Google's AI research division, Google AI, is at the forefront of developing cutting-edge AI technologies. From self-driving cars to natural language understanding, Google AI is driving innovation across various domains.
    
    ### IBM Watson
    IBM Watson is a cognitive computing platform that leverages AI to analyze vast amounts of data, extract insights, and make informed decisions. Watson's applications span healthcare, finance, retail, and cybersecurity.
    
    ### Amazon AI
    Amazon's AI initiatives focus on enhancing customer experiences through personalized recommendations, voice recognition, and supply chain optimization. Amazon AI powers Alexa, Amazon Go stores, and Amazon Web Services (AWS) AI services.
    
    ### Microsoft AI
    Microsoft's AI division is dedicated to democratizing AI and making it accessible to businesses and developers worldwide. Microsoft AI solutions include Azure AI, cognitive services, and AI-powered productivity tools.
    
    ### OpenAI
    OpenAI is a research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. OpenAI conducts cutting-edge research in AI safety, ethics, and the development of AGI systems.
    
    ## Noteworthy News in Artificial Intelligence
    
    Recent breakthroughs in AI research have showcased the potential of AI to solve complex problems and drive innovation. Industry partnerships and collaborations are accelerating the adoption of AI technologies across sectors, while regulatory updates and policies are shaping the future of AI development.
    
    In conclusion, Artificial Intelligence continues to shape the future of technology and society, with the latest trends and key players driving innovation and progress. As we navigate the evolving landscape of AI, it's essential to stay informed, engage in ethical discussions, and embrace the transformative power of AI for the betterment of humanity.
    
    *Subscribe to our newsletter for regular updates on AI trends and news. Join our upcoming webinar on Ethical AI Development or share your thoughts and experiences with AI in the comments section below.*[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Editor] Task output: # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Artificial Intelligence (AI) has revolutionized various industries with its capabilities to automate processes, analyze data, and make predictions. The rapid advancements in AI technology have paved the way for innovative solutions that enhance efficiency and drive growth. As we delve into the latest trends and key players in the AI industry, it's essential to understand the impact of AI on society and the ethical considerations that come with its development.
    
    ## Latest Trends in Artificial Intelligence
    
    ### Deep learning and neural networks
    Deep learning algorithms and neural networks have significantly improved AI's ability to learn from data, recognize patterns, and make decisions. These technologies have been instrumental in powering applications like image recognition, speech recognition, and autonomous vehicles.
    
    ### Natural language processing
    Natural language processing (NLP) enables machines to understand and interpret human language, facilitating communication between humans and computers. NLP applications include chatbots, language translation, sentiment analysis, and voice assistants like Siri and Alexa.
    
    ### AI-based automation and robotics
    AI-driven automation and robotics have transformed industries by streamlining processes, increasing productivity, and reducing human error. From manufacturing and logistics to healthcare and finance, AI-powered robots are reshaping the way work is done.
    
    ### AI in healthcare and medicine
    The healthcare industry has embraced AI to improve patient care, diagnosis accuracy, and treatment outcomes. AI applications in healthcare include medical imaging analysis, drug discovery, personalized medicine, and predictive analytics for disease prevention.
    
    ### Ethical considerations in AI development
    As AI technology continues to evolve, it's crucial to address ethical considerations related to data privacy, bias in AI algorithms, transparency in decision-making, and the impact of AI on jobs and society. Ethical AI development frameworks and guidelines are essential to ensure that AI benefits humanity responsibly.
    
    ## Key Players in the AI Industry
    
    ### Google AI
    Google's AI research division, Google AI, is at the forefront of developing cutting-edge AI technologies. From self-driving cars to natural language understanding, Google AI is driving innovation across various domains.
    
    ### IBM Watson
    IBM Watson is a cognitive computing platform that leverages AI to analyze vast amounts of data, extract insights, and make informed decisions. Watson's applications span healthcare, finance, retail, and cybersecurity.
    
    ### Amazon AI
    Amazon's AI initiatives focus on enhancing customer experiences through personalized recommendations, voice recognition, and supply chain optimization. Amazon AI powers Alexa, Amazon Go stores, and Amazon Web Services (AWS) AI services.
    
    ### Microsoft AI
    Microsoft's AI division is dedicated to democratizing AI and making it accessible to businesses and developers worldwide. Microsoft AI solutions include Azure AI, cognitive services, and AI-powered productivity tools.
    
    ### OpenAI
    OpenAI is a research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. OpenAI conducts cutting-edge research in AI safety, ethics, and the development of AGI systems.
    
    ## Noteworthy News in Artificial Intelligence
    
    Recent breakthroughs in AI research have showcased the potential of AI to solve complex problems and drive innovation. Industry partnerships and collaborations are accelerating the adoption of AI technologies across sectors, while regulatory updates and policies are shaping the future of AI development.
    
    In conclusion, Artificial Intelligence continues to shape the future of technology and society, with the latest trends and key players driving innovation and progress. As we navigate the evolving landscape of AI, it's essential to stay informed, engage in ethical discussions, and embrace the transformative power of AI for the betterment of humanity.
    
    *Subscribe to our newsletter for regular updates on AI trends and news. Join our upcoming webinar on Ethical AI Development or share your thoughts and experiences with AI in the comments section below.*
    
    [00m


- Display the results of your execution as markdown in the notebook.


```python
from IPython.display import Markdown
Markdown(result)
```




# The Future of Artificial Intelligence in Healthcare

As technology continues to advance, the integration of artificial intelligence (AI) in healthcare is becoming increasingly prevalent. AI has the potential to revolutionize the industry by improving patient care, streamlining processes, and enhancing diagnostic accuracy. Staying updated on the latest trends in AI is crucial for healthcare professionals, patients, and researchers alike to ensure they are leveraging the most innovative solutions available.

## Latest Trends in AI Healthcare

One of the most significant trends in AI healthcare is the development of machine learning algorithms for medical diagnostics. These algorithms can analyze vast amounts of data to assist in early disease detection and personalized treatment plans. Additionally, AI is being utilized in personalized medicine to tailor treatment plans to individual patients based on their unique genetic makeup and medical history. The growth of telemedicine and virtual health assistants is also transforming the way patients access healthcare services, making it more convenient and efficient.

## Key Players in the Industry

Several key players are leading the way in AI healthcare solutions. Google Health is at the forefront, leveraging AI to develop innovative tools for medical professionals and patients. IBM Watson Health is also a major player, focusing on healthcare analytics to improve patient outcomes and operational efficiency. Startups like Babylon Health are disrupting traditional healthcare models by offering virtual consultations and AI-driven healthcare services.

## Noteworthy News and Ethical Considerations

Recent developments in AI healthcare include FDA approvals for AI-based medical devices, indicating a growing acceptance of AI technologies in the industry. Partnerships between tech companies and healthcare providers are facilitating the integration of AI solutions into existing healthcare systems. However, ethical considerations and data privacy concerns remain prevalent, highlighting the importance of establishing robust guidelines for the ethical use of AI in healthcare applications.

In conclusion, the future of artificial intelligence in healthcare holds immense potential to transform the industry and improve patient outcomes. By staying informed on the latest trends, key players, and ethical considerations in AI healthcare, professionals and patients can make informed decisions about leveraging AI technologies for better healthcare delivery.

Remember to stay updated on AI advancements in healthcare, explore AI tools and resources, and participate in upcoming webinars or conferences to stay at the forefront of this rapidly evolving field.



## Try it Yourself

- Pass in a topic of your choice and see what the agents come up with!


```python
topic = "YOUR TOPIC HERE"
result = crew.kickoff(inputs={"topic": topic})
```

    [1m[95m [DEBUG]: == Working Agent: Content Planner[00m
    [1m[95m [INFO]: == Starting Task: 1. Prioritize the latest trends, key players, and noteworthy news on YOUR TOPIC HERE.
    2. Identify the target audience, considering their interests and pain points.
    3. Develop a detailed content outline including an introduction, key points, and a call to action.
    4. Include SEO keywords and relevant data or sources.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer: 
    
    Title: The Future of Artificial Intelligence in Healthcare
    
    Introduction:
    - Brief overview of artificial intelligence in healthcare
    - Importance of staying updated on the latest trends in AI
    - Preview of key players in the industry and recent developments
    
    Key Points:
    1. Latest Trends:
    - Advances in machine learning algorithms for medical diagnostics
    - Use of AI in personalized medicine and treatment plans
    - Growth of telemedicine and virtual health assistants
    
    2. Key Players:
    - Google Health and their AI-powered healthcare solutions
    - IBM Watson Health and their role in healthcare analytics
    - Startups like Babylon Health disrupting traditional healthcare models
    
    3. Noteworthy News:
    - FDA approvals for AI-based medical devices
    - Partnerships between tech companies and healthcare providers
    - Ethical considerations and data privacy concerns in AI healthcare applications
    
    Target Audience:
    - Healthcare professionals interested in technology integration
    - Patients looking for innovative healthcare solutions
    - Researchers and developers in the AI and healthcare industries
    
    Call to Action:
    - Encourage readers to stay informed on AI advancements in healthcare
    - Invite audience to explore AI tools and resources for healthcare professionals
    - Promote upcoming webinars or conferences on AI in healthcare
    
    SEO Keywords:
    - Artificial Intelligence in Healthcare
    - AI Trends in Medicine
    - Healthcare Technology Innovations
    - Machine Learning in Medical Diagnostics
    
    Resources:
    1. "The Top Trends in Artificial Intelligence for Healthcare" - Forbes
    2. "AI and the Future of Healthcare" - Harvard Business Review
    3. "Ethical Considerations in AI Healthcare Applications" - Journal of Medical Ethics
    
    By following this comprehensive content plan, the blog article on the future of artificial intelligence in healthcare will provide valuable insights for the target audience, incorporating the latest trends, key players, and noteworthy news in the industry.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Planner] Task output: Title: The Future of Artificial Intelligence in Healthcare
    
    Introduction:
    - Brief overview of artificial intelligence in healthcare
    - Importance of staying updated on the latest trends in AI
    - Preview of key players in the industry and recent developments
    
    Key Points:
    1. Latest Trends:
    - Advances in machine learning algorithms for medical diagnostics
    - Use of AI in personalized medicine and treatment plans
    - Growth of telemedicine and virtual health assistants
    
    2. Key Players:
    - Google Health and their AI-powered healthcare solutions
    - IBM Watson Health and their role in healthcare analytics
    - Startups like Babylon Health disrupting traditional healthcare models
    
    3. Noteworthy News:
    - FDA approvals for AI-based medical devices
    - Partnerships between tech companies and healthcare providers
    - Ethical considerations and data privacy concerns in AI healthcare applications
    
    Target Audience:
    - Healthcare professionals interested in technology integration
    - Patients looking for innovative healthcare solutions
    - Researchers and developers in the AI and healthcare industries
    
    Call to Action:
    - Encourage readers to stay informed on AI advancements in healthcare
    - Invite audience to explore AI tools and resources for healthcare professionals
    - Promote upcoming webinars or conferences on AI in healthcare
    
    SEO Keywords:
    - Artificial Intelligence in Healthcare
    - AI Trends in Medicine
    - Healthcare Technology Innovations
    - Machine Learning in Medical Diagnostics
    
    Resources:
    1. "The Top Trends in Artificial Intelligence for Healthcare" - Forbes
    2. "AI and the Future of Healthcare" - Harvard Business Review
    3. "Ethical Considerations in AI Healthcare Applications" - Journal of Medical Ethics
    
    By following this comprehensive content plan, the blog article on the future of artificial intelligence in healthcare will provide valuable insights for the target audience, incorporating the latest trends, key players, and noteworthy news in the industry.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Content Writer[00m
    [1m[95m [INFO]: == Starting Task: 1. Use the content plan to craft a compelling blog post on YOUR TOPIC HERE.
    2. Incorporate SEO keywords naturally.
    3. Sections/Subtitles are properly named in an engaging manner.
    4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
    5. Proofread for grammatical errors and alignment with the brand's voice.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer: 
    
    # The Future of Artificial Intelligence in Healthcare
    
    As technology continues to advance, the integration of artificial intelligence (AI) in healthcare is becoming increasingly prevalent. AI has the potential to revolutionize the industry by improving patient care, streamlining processes, and enhancing diagnostic accuracy. Staying updated on the latest trends in AI is crucial for healthcare professionals, patients, and researchers alike to ensure they are leveraging the most innovative solutions available.
    
    ## Latest Trends in AI Healthcare
    
    One of the most significant trends in AI healthcare is the development of machine learning algorithms for medical diagnostics. These algorithms can analyze vast amounts of data to assist in early disease detection and personalized treatment plans. Additionally, AI is being utilized in personalized medicine to tailor treatment plans to individual patients based on their unique genetic makeup and medical history. The growth of telemedicine and virtual health assistants is also transforming the way patients access healthcare services, making it more convenient and efficient.
    
    ## Key Players in the Industry
    
    Several key players are leading the way in AI healthcare solutions. Google Health is at the forefront, leveraging AI to develop innovative tools for medical professionals and patients. IBM Watson Health is also a major player, focusing on healthcare analytics to improve patient outcomes and operational efficiency. Startups like Babylon Health are disrupting traditional healthcare models by offering virtual consultations and AI-driven healthcare services.
    
    ## Noteworthy News and Ethical Considerations
    
    Recent developments in AI healthcare include FDA approvals for AI-based medical devices, indicating a growing acceptance of AI technologies in the industry. Partnerships between tech companies and healthcare providers are facilitating the integration of AI solutions into existing healthcare systems. However, ethical considerations and data privacy concerns remain prevalent, highlighting the importance of establishing robust guidelines for the ethical use of AI in healthcare applications.
    
    In conclusion, the future of artificial intelligence in healthcare holds immense potential to transform the industry and improve patient outcomes. By staying informed on the latest trends, key players, and ethical considerations in AI healthcare, professionals and patients can make informed decisions about leveraging AI technologies for better healthcare delivery.
    
    Remember to stay updated on AI advancements in healthcare, explore AI tools and resources, and participate in upcoming webinars or conferences to stay at the forefront of this rapidly evolving field.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Writer] Task output: # The Future of Artificial Intelligence in Healthcare
    
    As technology continues to advance, the integration of artificial intelligence (AI) in healthcare is becoming increasingly prevalent. AI has the potential to revolutionize the industry by improving patient care, streamlining processes, and enhancing diagnostic accuracy. Staying updated on the latest trends in AI is crucial for healthcare professionals, patients, and researchers alike to ensure they are leveraging the most innovative solutions available.
    
    ## Latest Trends in AI Healthcare
    
    One of the most significant trends in AI healthcare is the development of machine learning algorithms for medical diagnostics. These algorithms can analyze vast amounts of data to assist in early disease detection and personalized treatment plans. Additionally, AI is being utilized in personalized medicine to tailor treatment plans to individual patients based on their unique genetic makeup and medical history. The growth of telemedicine and virtual health assistants is also transforming the way patients access healthcare services, making it more convenient and efficient.
    
    ## Key Players in the Industry
    
    Several key players are leading the way in AI healthcare solutions. Google Health is at the forefront, leveraging AI to develop innovative tools for medical professionals and patients. IBM Watson Health is also a major player, focusing on healthcare analytics to improve patient outcomes and operational efficiency. Startups like Babylon Health are disrupting traditional healthcare models by offering virtual consultations and AI-driven healthcare services.
    
    ## Noteworthy News and Ethical Considerations
    
    Recent developments in AI healthcare include FDA approvals for AI-based medical devices, indicating a growing acceptance of AI technologies in the industry. Partnerships between tech companies and healthcare providers are facilitating the integration of AI solutions into existing healthcare systems. However, ethical considerations and data privacy concerns remain prevalent, highlighting the importance of establishing robust guidelines for the ethical use of AI in healthcare applications.
    
    In conclusion, the future of artificial intelligence in healthcare holds immense potential to transform the industry and improve patient outcomes. By staying informed on the latest trends, key players, and ethical considerations in AI healthcare, professionals and patients can make informed decisions about leveraging AI technologies for better healthcare delivery.
    
    Remember to stay updated on AI advancements in healthcare, explore AI tools and resources, and participate in upcoming webinars or conferences to stay at the forefront of this rapidly evolving field.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Editor[00m
    [1m[95m [INFO]: == Starting Task: Proofread the given blog post for grammatical errors and alignment with the brand's voice.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m


    [32;1m[1;3mI now can give a great answer
    
    Final Answer:
    # The Future of Artificial Intelligence in Healthcare
    
    As technology continues to advance, the integration of artificial intelligence (AI) in healthcare is becoming increasingly prevalent. AI has the potential to revolutionize the industry by improving patient care, streamlining processes, and enhancing diagnostic accuracy. Staying updated on the latest trends in AI is crucial for healthcare professionals, patients, and researchers alike to ensure they are leveraging the most innovative solutions available.
    
    ## Latest Trends in AI Healthcare
    
    One of the most significant trends in AI healthcare is the development of machine learning algorithms for medical diagnostics. These algorithms can analyze vast amounts of data to assist in early disease detection and personalized treatment plans. Additionally, AI is being utilized in personalized medicine to tailor treatment plans to individual patients based on their unique genetic makeup and medical history. The growth of telemedicine and virtual health assistants is also transforming the way patients access healthcare services, making it more convenient and efficient.
    
    ## Key Players in the Industry
    
    Several key players are leading the way in AI healthcare solutions. Google Health is at the forefront, leveraging AI to develop innovative tools for medical professionals and patients. IBM Watson Health is also a major player, focusing on healthcare analytics to improve patient outcomes and operational efficiency. Startups like Babylon Health are disrupting traditional healthcare models by offering virtual consultations and AI-driven healthcare services.
    
    ## Noteworthy News and Ethical Considerations
    
    Recent developments in AI healthcare include FDA approvals for AI-based medical devices, indicating a growing acceptance of AI technologies in the industry. Partnerships between tech companies and healthcare providers are facilitating the integration of AI solutions into existing healthcare systems. However, ethical considerations and data privacy concerns remain prevalent, highlighting the importance of establishing robust guidelines for the ethical use of AI in healthcare applications.
    
    In conclusion, the future of artificial intelligence in healthcare holds immense potential to transform the industry and improve patient outcomes. By staying informed on the latest trends, key players, and ethical considerations in AI healthcare, professionals and patients can make informed decisions about leveraging AI technologies for better healthcare delivery.
    
    Remember to stay updated on AI advancements in healthcare, explore AI tools and resources, and participate in upcoming webinars or conferences to stay at the forefront of this rapidly evolving field.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Editor] Task output: # The Future of Artificial Intelligence in Healthcare
    
    As technology continues to advance, the integration of artificial intelligence (AI) in healthcare is becoming increasingly prevalent. AI has the potential to revolutionize the industry by improving patient care, streamlining processes, and enhancing diagnostic accuracy. Staying updated on the latest trends in AI is crucial for healthcare professionals, patients, and researchers alike to ensure they are leveraging the most innovative solutions available.
    
    ## Latest Trends in AI Healthcare
    
    One of the most significant trends in AI healthcare is the development of machine learning algorithms for medical diagnostics. These algorithms can analyze vast amounts of data to assist in early disease detection and personalized treatment plans. Additionally, AI is being utilized in personalized medicine to tailor treatment plans to individual patients based on their unique genetic makeup and medical history. The growth of telemedicine and virtual health assistants is also transforming the way patients access healthcare services, making it more convenient and efficient.
    
    ## Key Players in the Industry
    
    Several key players are leading the way in AI healthcare solutions. Google Health is at the forefront, leveraging AI to develop innovative tools for medical professionals and patients. IBM Watson Health is also a major player, focusing on healthcare analytics to improve patient outcomes and operational efficiency. Startups like Babylon Health are disrupting traditional healthcare models by offering virtual consultations and AI-driven healthcare services.
    
    ## Noteworthy News and Ethical Considerations
    
    Recent developments in AI healthcare include FDA approvals for AI-based medical devices, indicating a growing acceptance of AI technologies in the industry. Partnerships between tech companies and healthcare providers are facilitating the integration of AI solutions into existing healthcare systems. However, ethical considerations and data privacy concerns remain prevalent, highlighting the importance of establishing robust guidelines for the ethical use of AI in healthcare applications.
    
    In conclusion, the future of artificial intelligence in healthcare holds immense potential to transform the industry and improve patient outcomes. By staying informed on the latest trends, key players, and ethical considerations in AI healthcare, professionals and patients can make informed decisions about leveraging AI technologies for better healthcare delivery.
    
    Remember to stay updated on AI advancements in healthcare, explore AI tools and resources, and participate in upcoming webinars or conferences to stay at the forefront of this rapidly evolving field.
    
    [00m



```python
Markdown(result)
```




# The Future of Artificial Intelligence in Healthcare

As technology continues to advance, the integration of artificial intelligence (AI) in healthcare is becoming increasingly prevalent. AI has the potential to revolutionize the industry by improving patient care, streamlining processes, and enhancing diagnostic accuracy. Staying updated on the latest trends in AI is crucial for healthcare professionals, patients, and researchers alike to ensure they are leveraging the most innovative solutions available.

## Latest Trends in AI Healthcare

One of the most significant trends in AI healthcare is the development of machine learning algorithms for medical diagnostics. These algorithms can analyze vast amounts of data to assist in early disease detection and personalized treatment plans. Additionally, AI is being utilized in personalized medicine to tailor treatment plans to individual patients based on their unique genetic makeup and medical history. The growth of telemedicine and virtual health assistants is also transforming the way patients access healthcare services, making it more convenient and efficient.

## Key Players in the Industry

Several key players are leading the way in AI healthcare solutions. Google Health is at the forefront, leveraging AI to develop innovative tools for medical professionals and patients. IBM Watson Health is also a major player, focusing on healthcare analytics to improve patient outcomes and operational efficiency. Startups like Babylon Health are disrupting traditional healthcare models by offering virtual consultations and AI-driven healthcare services.

## Noteworthy News and Ethical Considerations

Recent developments in AI healthcare include FDA approvals for AI-based medical devices, indicating a growing acceptance of AI technologies in the industry. Partnerships between tech companies and healthcare providers are facilitating the integration of AI solutions into existing healthcare systems. However, ethical considerations and data privacy concerns remain prevalent, highlighting the importance of establishing robust guidelines for the ethical use of AI in healthcare applications.

In conclusion, the future of artificial intelligence in healthcare holds immense potential to transform the industry and improve patient outcomes. By staying informed on the latest trends, key players, and ethical considerations in AI healthcare, professionals and patients can make informed decisions about leveraging AI technologies for better healthcare delivery.

Remember to stay updated on AI advancements in healthcare, explore AI tools and resources, and participate in upcoming webinars or conferences to stay at the forefront of this rapidly evolving field.



<a name='1'></a>
 ## Other Popular Models as LLM for your Agents

#### Hugging Face (HuggingFaceHub endpoint)

```Python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token="<HF_TOKEN_HERE>",
    task="text-generation",
)

### you will pass "llm" to your agent function
```

#### Mistral API

```Python
OPENAI_API_KEY=your-mistral-api-key
OPENAI_API_BASE=https://api.mistral.ai/v1
OPENAI_MODEL_NAME="mistral-small"
```

#### Cohere

```Python
from langchain_community.chat_models import ChatCohere
# Initialize language model
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
llm = ChatCohere()

### you will pass "llm" to your agent function
```

### For using Llama locally with Ollama and more, checkout the crewAI documentation on [Connecting to any LLM](https://docs.crewai.com/how-to/LLM-Connections/).


```python

```


```python

```


```python

```
