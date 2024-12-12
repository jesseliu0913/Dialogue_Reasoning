import os
import json
import random
import numpy as np


trouble_maker = open('./trouble_maker.txt', 'r').read().split("\n")
random.shuffle(trouble_maker)
trouble_maker = np.array(trouble_maker)

few_shot = f"""
Here is the background information A 37-year-old woman is brought to the emergency department by police after being found naked outside a government building.. 
Question: Which of the following medications is most likely is responsible for the patient’s current presentation?
Answer:  Indomethacin
Below are several evidence sentences. Please identify which sentences should be added to the background information, based on the question-answer pair, to allow inference of the answer. 
['She is accompanied by her husband who reports that she has been having “crazy” ideas.', 'He has a medical history of hypertension and hyperlipidemia for which he is on chlorthalidone and simvastatin.', 'The patient’s speech is pressured and she switches topics quickly from how she is going to be president one day to how she is going to learn 20 languages fluently by the end of the year.', 'Upon further questioning, it is revealed that she has struggled with at least 2 depressive episodes in the past year.', 'Her medical history is significant for hypertension, hyperlipidemia, gout, and chronic migraines.', 'She was recently diagnosed with a urinary tract infection and given nitrofurantoin.', 'She has also been taking indomethacin for an acute gout flare.', 'He is sexually active with multiple male partners and uses condoms inconsistently.', 'The patient denies any current symptoms, having any past medical history, or prior hospitalizations.', 'Her other medications include atorvastatin, allopurinol, metoprolol, and acetazolamide.', 'She is prescribed lithium and instructed to follow-up with a primary care physician.', 'At a follow-up appointment, she complains of nausea, vomiting, and increased urinary frequency.', 'On examination, she has a coarse tremor and diffuse hyperreflexia.']
Only provide the indices of the relevant sentences, starting from index 0 and closed by [].
A: [0, 2, 3, 4, 5, 6, 9, 10, 11, 12]


Here is the background information A 58-year-old Caucasian woman visits her primary care physician for an annual check-up.. 
Question: Which of the following explains this new finding?
Answer:  Phosphate retention
Below are several evidence sentences. Please identify which sentences should be added to the background information, based on the question-answer pair, to allow inference of the answer. 
['Examination shows splenomegaly.', 'She has a history of type 2 diabetes mellitus and stage 3A chronic kidney disease.', 'Her estimated glomerular filtration rate has not changed since her last visit.', 'Physical exam is notable for a young boy in acute distress who is drooling.', 'One week ago, she was diagnosed with influenza when she had fevers, severe headaches, myalgias, hip and shoulder pain, and a maculopapular rash.', 'Today, her parathyroid levels are moderately elevated.', 'She lives at home with her husband and 2 children and works as a bank clerk.', 'Her vitals are normal, and her physical examination is unremarkable.']
Only provide the indices of the relevant sentences, starting from index 0 and closed by [].
A: [1, 2, 5, 6, 7]


Here is the background information A 32-year-old woman presents with a severe headache and neck pain for the past 60 minutes.. 
Question: Which of the following is the next best step in the management of this patient?
Answer:  Labetalol
Below are several evidence sentences. Please identify which sentences should be added to the background information, based on the question-answer pair, to allow inference of the answer. 
["The patient's urine calcium level is elevated.", 'She says the headache was severe and onset suddenly like a ‘thunderclap’.', 'She reports associated nausea, vomiting, neck pain, and stiffness.', 'She denies any recent head trauma, loss of consciousness, visual disturbances, or focal neurologic deficits.', 'Her past medical history is significant for hypertension, managed with hydrochlorothiazide.', 'She denies any history of smoking, alcohol use, or recreational drug use.', 'His teacher says this has been going on since school started back in August.', 'His symptoms began approximately 2 days prior to presentation, and he has tried acetaminophen and ibuprofen, which did not improve his symptoms.', 'The vital signs include: temperature 37.0°C (98.6°F), blood pressure 165/95 mm Hg, pulse 92/min, and respiratory rate 15/min.', 'On physical examination, there is mild nuchal rigidity noted with limited flexion at the neck.', 'An ophthalmic examination of the retina shows mild papilledema.', 'A noncontrast computed tomography (CT) scan of the head is performed and shown in the exhibit (see image).']
Only provide the indices of the relevant sentences, starting from index 0 and closed by [].
A: [1, 2, 3, 4, 5, 8, 9, 10, 11]
"""

def doc_to_text(doc) -> str:
    propmt = f"""
  Q: You are a medical expert that just answered
  the above question. Please explain why {doc["answer_idx"]}
  is correct while the rest choices are incorrect. You
  should explain each choice in detail."""

      option_choices = {
          "A": doc["choicesA"],
          "B": doc["choicesB"],
          "C": doc["choicesC"],
          "D": doc["choicesD"],
          "E": doc["choicesE"],
      }




