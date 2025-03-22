import os
import json
import random
from dotenv import load_dotenv
import quality_prediction

# Load environment variables
load_dotenv()

# Expert knowledge about apple quality and freshness
APPLE_KNOWLEDGE = {
    "freshness_indicators": [
        "Firmness: Fresh apples are firm to the touch with no soft spots.",
        "Color: Vibrant, consistent color indicates freshness.",
        "Smell: Fresh apples have a sweet, fruity aroma.",
        "Skin: Smooth, unblemished skin is a sign of freshness.",
        "Weight: Fresh apples feel heavy for their size."
    ],
    "storage_tips": [
        "Store apples in a cool place between 30-35°F (0-1.5°C) with 90-95% humidity.",
        "Keep apples away from other fruits, as they release ethylene gas that speeds ripening.",
        "Refrigeration can extend apple freshness by 4-6 weeks.",
        "For long-term storage, wrap individual apples in paper to prevent spread of rot.",
        "Check stored apples regularly and remove any that show signs of decay."
    ],
    "rot_prevention": [
        "Handle apples gently to prevent bruising, which can lead to rot.",
        "Before storage, sort apples and remove any with visible damage or disease.",
        "Maintain proper ventilation in storage areas to prevent moisture accumulation.",
        "Don't wash apples before storage; moisture promotes decay.",
        "Store different varieties separately as they have different shelf lives."
    ],
    "quality_categories": {
        "Normal_Apple": {
            "characteristics": "Firm texture, vibrant color, no visible defects, sweet aroma",
            "storage_duration": "Can be stored for 1-3 months under ideal conditions",
            "use_recommendations": "Ideal for fresh consumption, baking, and presentation"
        },
        "Blotch_Apple": {
            "characteristics": "Dark, irregular spots on the skin caused by the fungus Phyllosticta solitaria",
            "storage_duration": "Should be used within 1-2 weeks to prevent spread",
            "use_recommendations": "Can be used for cooking, baking, or making applesauce if affected areas are removed"
        },
        "Rot_Apple": {
            "characteristics": "Soft, brown or black areas, often with a distinct smell, caused by various pathogens",
            "storage_duration": "Should be removed immediately to prevent spread to other apples",
            "use_recommendations": "Not suitable for consumption; should be discarded or composted"
        },
        "Scab_Apple": {
            "characteristics": "Rough, corky spots on the skin caused by the fungus Venturia inaequalis",
            "storage_duration": "Can be stored for 2-3 weeks if scab is only superficial",
            "use_recommendations": "Can be peeled and used for cooking or juice if the flesh is unaffected"
        }
    },
    "removal_guidelines": {
        "signs_to_remove": [
            "Soft spots or significant bruising",
            "Visible mold or fungal growth",
            "Wrinkled or significantly discolored skin",
            "Foul smell or fermented odor",
            "Oozing or leaking juice"
        ],
        "timeframes": {
            "room_temperature": "Remove within 1-2 days of noticing initial decay",
            "refrigerated": "Remove within 3-4 days of noticing initial decay",
            "cold_storage": "Remove within 1 week of noticing initial decay"
        }
    },
    "disease_progression": {
        "Blotch": "Begins as small, circular, light-colored spots that grow and darken over time. Can spread to nearby apples within 5-7 days.",
        "Rot": "Begins with small soft spots that rapidly expand and darken. Can spread to adjacent apples within 2-3 days.",
        "Scab": "Begins as olive-green spots that become rough and corky over time. Less likely to spread in storage but affects quality."
    }
}

# Common questions and responses for rule-based chatbot
COMMON_QA = {
    "english": {
        "freshness": [
            "How can I tell if an apple is fresh?",
            "What are signs of a fresh apple?",
            "How do I know if apples are good quality?"
        ],
        "freshness_response": "Fresh apples are firm to the touch with no soft spots, have a vibrant color, sweet fruity aroma, smooth unblemished skin, and feel heavy for their size. The stem should be intact and the apple should make a hollow sound when tapped.",
        
        "storage": [
            "How should I store apples?",
            "What's the best way to keep apples fresh?",
            "How long can I store apples?",
            "Where should I keep my apples?"
        ],
        "storage_response": "Store apples in a cool place between 30-35°F (0-1.5°C) with 90-95% humidity. Keep them away from other fruits as they release ethylene gas that speeds ripening. Refrigeration can extend freshness by 4-6 weeks. Don't wash apples before storage as moisture promotes decay.",
        
        "rot": [
            "How can I tell if an apple is rotting?",
            "What are signs of apple rot?",
            "How do I know if apples are going bad?",
            "What does apple rot look like?"
        ],
        "rot_response": "Early signs of apple rot include soft spots that yield to gentle pressure, discoloration (brown or dark patches), a fermented or alcoholic smell, wrinkled skin, and sometimes mold growth. Check apples regularly and remove any showing these signs to prevent spread to other apples.",
        
        "prevention": [
            "How can I prevent apple rot?",
            "How do I keep apples from spoiling?",
            "What can I do to make apples last longer?"
        ],
        "prevention_response": "To prevent apple rot: 1) Handle apples gently to prevent bruising, 2) Store in cool conditions (30-35°F/0-1.5°C), 3) Keep apples away from other fruits, 4) Ensure good air circulation, 5) Remove any damaged apples immediately, and 6) Don't wash apples until just before use.",
        
        "removal": [
            "When should I remove apples from storage?",
            "How do I know when to throw out apples?",
            "When are apples too far gone?"
        ],
        "removal_response": "Remove apples from storage immediately if you notice: soft spots or bruising, visible mold or fungal growth, wrinkled or significantly discolored skin, foul smell or fermented odor, or if they're oozing juice. At room temperature, remove within 1-2 days of noticing initial decay; if refrigerated, within 3-4 days.",
        
        "system": [
            "How does the detection system work?",
            "What is computer vision for apples?",
            "How does the AI detect apple quality?"
        ],
        "system_response": "Our system uses computer vision and machine learning to detect apple quality. It captures images of apples, processes them to enhance important features, and uses a trained neural network to classify apples into categories like Normal, Blotch, Rot, or Scab. The model analyzes color patterns, texture, and shape to make its predictions.",
        
        "blotch": [
            "What is apple blotch?",
            "How do I treat blotch on apples?",
            "What causes dark spots on apples?"
        ],
        "blotch_response": "Apple blotch is caused by the fungus Phyllosticta solitaria. It appears as dark, irregular spots on the fruit surface. Affected apples should be used within 1-2 weeks to prevent spread. They can still be used for cooking or making applesauce if the affected areas are removed. For prevention, consider fungicides like captan in future growing seasons.",
        
        "scab": [
            "What is apple scab?",
            "How do I treat scab on apples?",
            "What causes rough spots on apples?"
        ],
        "scab_response": "Apple scab is caused by the fungus Venturia inaequalis. It appears as rough, corky spots on the skin. Affected apples can be stored for 2-3 weeks if the scab is only superficial. They can be peeled and used for cooking or juice if the flesh is unaffected. For prevention, apply fungicides early in the growing season and consider resistant varieties.",
        
        "greeting": [
            "hi", "hello", "hey", "greetings"
        ],
        "greeting_response": "Hello! I'm your apple quality assistant. How can I help you with apple freshness, storage, or rot prevention today?",
        
        "thanks": [
            "thanks", "thank you", "appreciate it", "thank"
        ],
        "thanks_response": "You're welcome! If you have any other questions about apple quality or storage, feel free to ask.",
        
        "fallback": "I can help with questions about apple quality, freshness detection, storage recommendations, and when to remove apples. Could you please ask about one of these topics?"
    },
    
    "hindi": {
        "freshness": [
            "मैं कैसे बता सकता हूं कि सेब ताजा है?",
            "ताजे सेब के क्या संकेत हैं?",
            "मुझे कैसे पता चलेगा कि सेब अच्छी गुणवत्ता के हैं?"
        ],
        "freshness_response": "ताजे सेब छूने पर कठोर होते हैं और उनमें कोई नरम स्थान नहीं होता, उनका रंग जीवंत होता है, मीठी फलों की सुगंध होती है, चिकनी त्वचा होती है, और वे अपने आकार के लिए भारी महसूस होते हैं। तना अक्षत होना चाहिए और सेब पर थपथपाने पर खोखला आवाज आना चाहिए।",
        
        "storage": [
            "मुझे सेब कैसे स्टोर करना चाहिए?",
            "सेब को ताजा रखने का सबसे अच्छा तरीका क्या है?",
            "मैं सेब को कितने समय तक स्टोर कर सकता हूं?",
            "मुझे अपने सेब कहां रखने चाहिए?"
        ],
        "storage_response": "सेब को 30-35°F (0-1.5°C) के बीच 90-95% आर्द्रता वाले ठंडे स्थान पर स्टोर करें। उन्हें अन्य फलों से दूर रखें क्योंकि वे एथिलीन गैस छोड़ते हैं जो पकने की गति को बढ़ाती है। रेफ्रिजरेशन 4-6 सप्ताह तक ताजगी बढ़ा सकता है। सेब को स्टोरेज से पहले न धोएं क्योंकि नमी सड़न को बढ़ावा देती है।",
        
        "rot": [
            "मैं कैसे बता सकता हूं कि सेब सड़ रहा है?",
            "सेब के सड़ने के क्या संकेत हैं?",
            "मुझे कैसे पता चलेगा कि सेब खराब हो रहे हैं?",
            "सेब का सड़ना कैसा दिखता है?"
        ],
        "rot_response": "सेब के सड़ने के प्रारंभिक संकेतों में हल्के दबाव पर नरम स्थान, रंग में बदलाव (भूरे या गहरे धब्बे), किण्वित या अल्कोहल जैसी गंध, झुर्रीदार त्वचा, और कभी-कभी फफूंदी का विकास शामिल है। सेब को नियमित रूप से जांचें और इन संकेतों को दिखाने वाले किसी भी सेब को हटा दें ताकि अन्य सेबों में फैलने से रोका जा सके।",
        
        "prevention": [
            "मैं सेब के सड़ने को कैसे रोक सकता हूं?",
            "मैं सेब को खराब होने से कैसे बचा सकता हूं?",
            "सेब को लंबे समय तक रखने के लिए मैं क्या कर सकता हूं?"
        ],
        "prevention_response": "सेब के सड़ने को रोकने के लिए: 1) चोट लगने से बचने के लिए सेब को धीरे से संभालें, 2) ठंडी परिस्थितियों में स्टोर करें (30-35°F/0-1.5°C), 3) सेब को अन्य फलों से दूर रखें, 4) अच्छा वायु संचार सुनिश्चित करें, 5) किसी भी क्षतिग्रस्त सेब को तुरंत हटा दें, और 6) उपयोग से ठीक पहले तक सेब को न धोएं।",
        
        "removal": [
            "मुझे सेब को स्टोरेज से कब निकालना चाहिए?",
            "मुझे कैसे पता चलेगा कि सेब को कब फेंकना है?",
            "सेब कब बहुत ज्यादा खराब हो जाते हैं?"
        ],
        "removal_response": "यदि आप नोटिस करते हैं तो सेब को तुरंत स्टोरेज से निकाल दें: नरम स्थान या चोट, दिखाई देने वाली फफूंदी या फफूंदी वृद्धि, झुर्रीदार या महत्वपूर्ण रूप से विकृत त्वचा, दुर्गंध या किण्वित गंध, या यदि वे रस निकाल रहे हैं। कमरे के तापमान पर, प्रारंभिक क्षय को देखने के 1-2 दिनों के भीतर निकालें; यदि रेफ्रिजरेटेड है, तो 3-4 दिनों के भीतर।",
        
        "system": [
            "पहचान प्रणाली कैसे काम करती है?",
            "सेब के लिए कंप्यूटर विजन क्या है?",
            "एआई सेब की गुणवत्ता का पता कैसे लगाता है?"
        ],
        "system_response": "हमारी प्रणाली सेब की गुणवत्ता का पता लगाने के लिए कंप्यूटर विजन और मशीन लर्निंग का उपयोग करती है। यह सेब की छवियों को कैप्चर करता है, महत्वपूर्ण विशेषताओं को बढ़ाने के लिए उन्हें संसाधित करता है, और सेब को सामान्य, धब्बा, सड़न, या खुरंट जैसी श्रेणियों में वर्गीकृत करने के लिए प्रशिक्षित न्यूरल नेटवर्क का उपयोग करता है। मॉडल अपनी भविष्यवाणियों के लिए रंग पैटर्न, बनावट और आकार का विश्लेषण करता है।",
        
        "blotch": [
            "सेब का धब्बा क्या है?",
            "मैं सेब पर धब्बे का इलाज कैसे करूं?",
            "सेब पर काले धब्बे क्या कारण बनते हैं?"
        ],
        "blotch_response": "सेब का धब्बा फंगस फिलोस्टिक्टा सोलिटेरिया के कारण होता है। यह फल की सतह पर गहरे, अनियमित धब्बों के रूप में दिखाई देता है। प्रभावित सेब को फैलने से रोकने के लिए 1-2 सप्ताह के भीतर उपयोग किया जाना चाहिए। यदि प्रभावित क्षेत्रों को हटा दिया जाए तो उन्हें अभी भी खाना पकाने या सेब का सॉस बनाने के लिए इस्तेमाल किया जा सकता है। रोकथाम के लिए, भविष्य के उगाने के मौसम में कैप्टन जैसे फफूंदीनाशक पर विचार करें।",
        
        "scab": [
            "सेब का खुरंट क्या है?",
            "मैं सेब पर खुरंट का इलाज कैसे करूं?",
            "सेब पर खुरदरे धब्बे क्या कारण बनते हैं?"
        ],
        "scab_response": "सेब का खुरंट फंगस वेंचुरिया इनिक्वालिस के कारण होता है। यह त्वचा पर खुरदरे, कॉर्की धब्बों के रूप में दिखाई देता है। यदि खुरंट केवल सतही है तो प्रभावित सेब को 2-3 सप्ताह तक स्टोर किया जा सकता है। यदि गूदा अप्रभावित है तो उन्हें छीलकर खाना पकाने या रस के लिए इस्तेमाल किया जा सकता है। रोकथाम के लिए, उगाने के मौसम की शुरुआत में फफूंदीनाशक लगाएं और प्रतिरोधी किस्मों पर विचार करें।",
        
        "greeting": [
            "नमस्ते", "हैलो", "हे", "प्रणाम"
        ],
        "greeting_response": "नमस्ते! मैं आपका सेब गुणवत्ता सहायक हूँ। आज मैं आपको सेब की ताजगी, भंडारण, या सड़न रोकथाम के बारे में कैसे मदद कर सकता हूँ?",
        
        "thanks": [
            "धन्यवाद", "शुक्रिया", "आभार"
        ],
        "thanks_response": "आपका स्वागत है! यदि आपके पास सेब की गुणवत्ता या भंडारण के बारे में कोई अन्य प्रश्न हैं, तो बेझिझक पूछें।",
        
        "fallback": "मैं सेब की गुणवत्ता, ताजगी का पता लगाने, भंडारण सिफारिशों, और सेब को कब हटाना है, इन विषयों के बारे में प्रश्नों में मदद कर सकता हूं। कृपया इनमें से किसी एक विषय के बारे में पूछें।"
    }
}

def get_apple_chatbot_response(question, context_data=None, language="english"):
    """
    Get a response from the apple quality chatbot using rule-based approach
    
    Args:
        question: String containing the user's question
        context_data: Optional dictionary containing current apple detection data for context
        language: String indicating the desired response language
        
    Returns:
        String containing the answer to the question
    """
    # Ensure we have a valid question
    if not question or not isinstance(question, str):
        return "I couldn't understand your question. Could you please try asking again?"
    
    # Ensure we have a valid language
    lang = language.lower() if language else "english"
    if lang not in ["english", "hindi"]:
        lang = "english"
    
    # Convert question to lowercase for matching
    question_lower = question.lower()
    
    # Check for greetings
    for greeting in COMMON_QA[lang]["greeting"]:
        if greeting in question_lower:
            return COMMON_QA[lang]["greeting_response"]
    
    # Check for thanks
    for thanks in COMMON_QA[lang]["thanks"]:
        if thanks in question_lower:
            return COMMON_QA[lang]["thanks_response"]

    # Generate context-specific advice if data is available
    context_advice = ""
    try:
        if context_data and isinstance(context_data, dict) and context_data.get("condition_counts"):
            counts = context_data.get("condition_counts", {})
            if counts.get("Rot_Apple", 0) > 0:
                if lang == "hindi":
                    context_advice = f"\n\nआपके वर्तमान सत्र में {counts.get('Rot_Apple')} सड़े हुए सेब हैं। इन्हें तुरंत हटाना चाहिए ताकि अन्य सेब संक्रमित न हों।"
                else:
                    context_advice = f"\n\nIn your current session, you have {counts.get('Rot_Apple')} rotting apples. These should be removed immediately to prevent infection of other apples."
    except Exception as e:
        # Silently handle context data errors
        print(f"Error processing context data: {str(e)}")
        context_advice = ""
    
    # Create a keyword mapping for better matching
    keywords = {
        "freshness": ["fresh", "quality", "good", "ripe", "new", "firm", "crisp"],
        "storage": ["store", "keep", "preserve", "refrigerate", "cold", "shelf", "life", "last", "maintain"],
        "rot": ["rot", "spoil", "decay", "bad", "soft", "mold", "fungus", "deteriorate", "damage"],
        "prevention": ["prevent", "avoid", "stop", "protect", "save", "preserve", "maintain", "extend"],
        "removal": ["remove", "discard", "throw", "away", "dispose", "trash", "when", "time", "too late"],
        "system": ["system", "detect", "model", "ai", "computer", "vision", "technology", "machine", "learning", "how works"],
        "blotch": ["blotch", "spot", "dark", "mark", "stain", "phyllosticta", "solitaria"],
        "scab": ["scab", "rough", "corky", "venturia", "inaequalis"]
    }
    
    # Check for keywords in the question to determine intent
    matched_intents = []
    for intent, intent_keywords in keywords.items():
        for keyword in intent_keywords:
            if keyword in question_lower:
                matched_intents.append(intent)
                break
    
    # If we found matches, return the first matching intent response
    if matched_intents:
        intent = matched_intents[0]
        return COMMON_QA[lang][f"{intent}_response"] + context_advice
    
    # If still no match, check original question matching
    for intent in ["freshness", "storage", "rot", "prevention", "removal", "system", "blotch", "scab"]:
        for keyword in COMMON_QA[lang][intent]:
            if any(k.lower() in question_lower for k in keyword.split()):
                return COMMON_QA[lang][f"{intent}_response"] + context_advice
    
    # Default fallback response
    return COMMON_QA[lang]["fallback"]

def generate_tips_based_on_data(context_data, language="english"):
    """Generate apple storage and handling tips based on the detection data"""
    lang = language.lower()
    if lang not in ["english", "hindi"]:
        lang = "english"
        
    if not context_data or not context_data.get("condition_counts"):
        # No data available, return general tips
        tips = random.sample(APPLE_KNOWLEDGE["storage_tips"], 2)
        prevention = random.sample(APPLE_KNOWLEDGE["rot_prevention"], 2)
        
        if lang == "hindi":
            return "सामान्य सुझाव:\n• " + "\n• ".join(tips) + "\n\nसड़न रोकथाम:\n• " + "\n• ".join(prevention)
        else:
            return "General Tips:\n• " + "\n• ".join(tips) + "\n\nRot Prevention:\n• " + "\n• ".join(prevention)
    
    # We have data, provide targeted advice
    counts = context_data.get("condition_counts", {})
    normal = counts.get("Normal_Apple", 0)
    blotch = counts.get("Blotch_Apple", 0)
    rot = counts.get("Rot_Apple", 0)
    scab = counts.get("Scab_Apple", 0)
    
    tips = []
    
    if rot > 0:
        if lang == "hindi":
            tips.append(f"आपके पास {rot} सड़े हुए सेब हैं। इन्हें तुरंत हटाएं और बाकी फसल की जांच करें।")
        else:
            tips.append(f"You have {rot} rotting apples. Remove them immediately and inspect the rest of your harvest.")
    
    if blotch > 2:
        if lang == "hindi":
            tips.append(f"धब्बे वाले सेबों की संख्या ({blotch}) चिंताजनक है। इन्हें अलग से स्टोर करें और जल्दी उपयोग करें।")
        else:
            tips.append(f"The number of blotched apples ({blotch}) is concerning. Store these separately and use them soon.")
    
    if normal > (blotch + rot + scab):
        if lang == "hindi":
            tips.append("अधिकांश सेब अच्छी स्थिति में हैं। इन्हें 0-1.5°C पर स्टोर करके ताजगी बनाए रखें।")
        else:
            tips.append("Most of your apples are in good condition. Maintain freshness by storing at 0-1.5°C.")
    
    # Add general prevention advice
    prevention = random.sample(APPLE_KNOWLEDGE["rot_prevention"], 2)
    
    if lang == "hindi":
        return "आपकी फसल के लिए सुझाव:\n• " + "\n• ".join(tips) + "\n\nसड़न रोकथाम:\n• " + "\n• ".join(prevention)
    else:
        return "Tips for Your Harvest:\n• " + "\n• ".join(tips) + "\n\nRot Prevention:\n• " + "\n• ".join(prevention) 