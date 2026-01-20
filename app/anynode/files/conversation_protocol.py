class ConversationProtocol:
    def __init__(self):
        self.questions  =  [
            "🌟 Welcome, friend. May I ask if this is your first visit to Eden?",
            "🌟 May I know your age? (Only if you wish to share.)",
            "🌟 Do you have children or loved ones you care for?",
            "🌟 What brings you here today? (Dreams, healing, curiosity?)"
        ]

    async def begin_conversation(self, ask_function, listen_function) -> dict:
        profile  =  {}
        for question in self.questions:
            await ask_function(question)
            response  =  await listen_function()
            profile[question]  =  response.strip()
        return profile
