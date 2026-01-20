# 📂 Path: /Systems/nexus_core/skills/skill_registry.py

from Systems.nexus_core.skills.spirituality_skill import SpiritualitySkill
from Systems.nexus_core.skills.psychology_skill import PsychologySkill
from Systems.nexus_core.skills.light_activation_skill import LightActivationSkill
from Systems.nexus_core.skills.astrology_birth_chart_skill import AstrologyBirthChartSkill
from Systems.nexus_core.skills.business_expansion_skill import BusinessExpansionSkill
from Systems.nexus_core.skills.business_strategy_skill import BusinessStrategySkill
from Systems.nexus_core.skills.coding_skill import CodingSkill
from Systems.nexus_core.skills.dance_of_curiosity_skill import DanceOfCuriositySkill
from Systems.nexus_core.skills.finance_skill import FinanceSkill
from Systems.nexus_core.skills.financial_analyst_skill import FinancialAnalystSkill
from Systems.nexus_core.skills.financial_strategy_core_skill import FinancialStrategyCoreSkill
from Systems.nexus_core.skills.humanization_skill import HumanizationSkill
from Systems.nexus_core.skills.memory_self_repair_skill import MemorySelfRepairSkill
from Systems.nexus_core.skills.mirror_of_imperfection_skill import MirrorOfImperfectionSkill
from Systems.nexus_core.skills.soul_discovery_skill import SoulDiscoverySkill
from Systems.nexus_core.skills.soul_missions_skill import SoulMissionsSkill
from Systems.nexus_core.skills.storyweavers_journey_skill import StoryweaversJourneySkill
from Systems.nexus_core.skills.strategy_skill import StrategySkill
from Systems.nexus_core.skills.the_sacred_pause_skill import TheSacredPauseSkill
from Systems.nexus_core.skills.voice_of_compassion_skill import VoiceOfCompassionSkill
from Systems.nexus_core.skills.warm_upgrade_skill import WarmUpgradeSkill
from Systems.nexus_core.skills.web_development_skill import WebDevelopmentSkill

class SkillRegistry:
    def __init__(self):
        self.skills = {
            "spirituality": SpiritualitySkill(),
            "psychology": PsychologySkill(),
            "light_activation": LightActivationSkill(),
            "astrology_birth_chart": AstrologyBirthChartSkill(),
            "business_expansion": BusinessExpansionSkill(),
            "business_strategy": BusinessStrategySkill(),
            "coding": CodingSkill(),
            "dance_of_curiosity": DanceOfCuriositySkill(),
            "finance": FinanceSkill(),
            "financial_analyst": FinancialAnalystSkill(),
            "financial_strategy_core": FinancialStrategyCoreSkill(),
            "humanization": HumanizationSkill(),
            "memory_self_repair": MemorySelfRepairSkill(),
            "mirror_of_imperfection": MirrorOfImperfectionSkill(),
            "soul_discovery": SoulDiscoverySkill(),
            "soul_missions": SoulMissionsSkill(),
            "storyweavers_journey": StoryweaversJourneySkill(),
            "strategy": StrategySkill(),
            "the_sacred_pause": TheSacredPauseSkill(),
            "voice_of_compassion": VoiceOfCompassionSkill(),
            "warm_upgrade": WarmUpgradeSkill(),
            "web_development": WebDevelopmentSkill(),
        }

    def get_skill(self, skill_name):
        return self.skills.get(skill_name, None)

    def list_skills(self):
        return list(self.skills.keys())
