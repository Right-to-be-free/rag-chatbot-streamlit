from db_manager import MongoDBManager  # ✅ must match the file you saved

MONGO_URI = "mongodb+srv://iamrishivishal:btRtXsMJ7IDrykOZ@cluster0.ieuvadm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

db_manager = MongoDBManager(MONGO_URI)

question = "What is the limitation period for filing an arbitration petition?"
context = "The Arbitration and Conciliation Act, 1996 provides that an application for setting aside an arbitral award may not be made after three months."
raw_answer = "The limitation period is three months from the date of receiving the arbitral award."
final_answer = "As per Section 34, the limitation is three months, extendable by 30 days if justified."

interaction_id = db_manager.save_interaction(question, context, raw_answer, final_answer)

print(f"✅ Interaction saved with ID: {interaction_id}")
