from flask import Flask, request, jsonify, render_template,send_from_directory
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Initialize Flask app
app = Flask(__name__)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents
loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create a vectorstore with Chroma
vectorstore = Chroma.from_documents(docs, embeddings)

# Initialize the model
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Setup RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
@app.route("/")
def home():
    return render_template('index.html')  # Serve the homepage

@app.route("/template")
def chat_page():
    return render_template('template.html')  # Serve the chat page

@app.route("/material")
def material_page():
    return render_template('material.html')  # Serve the study material page

@app.route("/previous_questions")
def previous_questions_page():
    return render_template('previous_questions.html')  # Serve the previous questions page


#s1

@app.route("/material/S1/s1subgect")
def s1_subject():
    return render_template('material/S1/s1subgect.html')

@app.route("/material/S1/maths/s1")
def s1_maths():
    return render_template('material/S1/maths/s1.html')

@app.route("/material/S1/maths/<filename>")
def s1_maths_pdf(filename):
    return send_from_directory("static/material/S1/maths", filename)

@app.route("/material/S1/Engineering Physics A   PHT100/s1")
def s1_physics():
    return render_template('material/S1/Engineering Physics A   PHT100/s1.html')

@app.route("/material/S1/Engineering Physics A   PHT100/<filename>")
def s1_physics_pdf(filename):
    return send_from_directory("static/material/S1/Engineering Physics A   PHT100", filename)

@app.route("/material/S1/Engineering Graphics   EST110/graphics")
def s1_graphics():
    return render_template('material/S1/Engineering Graphics   EST110/graphics.html')

#graphics download and open
@app.route("/material/S1/Engineering Graphics   EST110/<filename>")#Engineering Graphics   EST110
def s1_engineering_graphics_pdf(filename):
    print(f"Serving file: static/material/S1/Engineering Graphics   EST110/{filename}")
    return send_from_directory("static/material/S1/Engineering Graphics   EST110", filename)

@app.route("/material/S1/s1(civil&mech)/s1")
def s1_civil_mech():
    return render_template('material/S1/s1(civil&mech)/s1.html')

@app.route("/material/S1/s1(civil&mech)/<filename>")
def s1_civil_mech_pdf(filename):
    return send_from_directory("static/material/S1/s1(civil&mech)", filename)

@app.route("/material/S1/Life Skills   HUN101/lifeskill")
def s1_lifeskills():
    return render_template('material/S1/Life Skills   HUN101/lifeskill.html')

@app.route("/material/S1/Life Skills   HUN101/<filename>")
def s1_lifeskills_pdf(filename):
    return send_from_directory("static/material/S1/Life Skills   HUN101", filename)


#S2

@app.route("/material/S2/s2")
def s2():
    return render_template('material/S2/s2.html')

# Semester 2 eee
@app.route("/material/S2/Basics Of Electrical & Electronics Engineering   EST130/s1")
def s2_est130():
    return render_template('material/S2/Basics Of Electrical & Electronics Engineering   EST130/s1.html')

@app.route("/material/S2/Basics Of Electrical & Electronics Engineering   EST130/<filename>")
def s2_est130_pdf(filename):
    return send_from_directory("static/material/S2/Basics Of Electrical & Electronics Engineering   EST130", filename)


#C Programming   EST102
@app.route("/material/S2/C Programming   EST102/s1")
def s2_est102():
    return render_template('material/S2/C Programming   EST102/s1.html')
@app.route("/material/S2/C Programming   EST102/<filename>")
def s2_est102_pdf(filename):
    return send_from_directory("static/material/S2/C Programming   EST102", filename)

#Engineering Chemistry   CYT100
@app.route("/material/S2/Engineering Chemistry   CYT100/s1")
def s2_cyt100():
    return render_template('material/S2/Engineering Chemistry   CYT100/s1.html')
@app.route("/material/S2/Engineering Chemistry   CYT100/<filename>")
def s2_cyt100_pdf(filename):
    return send_from_directory("static/material/S2/Engineering Chemistry   CYT100", filename)

#Engineering Mechanics   EST100
@app.route("/material/S2/Engineering Mechanics   EST100/s1")
def s2_est100():
    return render_template('material/S2/Engineering Mechanics   EST100/s1.html')

@app.route("/material/S2/Engineering Mechanics   EST100/<filename>")
def s2_est100_pdf(filename):
    return send_from_directory("static/material/S2/Engineering Mechanics   EST100", filename)

#Professional communication   HUN 102
@app.route("/material/S2/Professional communication   HUN 102/s1")
def s2_hun102():
    return render_template('material/S2/Professional communication   HUN 102/s1.html')

@app.route("/material/S2/Professional communication   HUN 102/<filename>")
def s2_hun102_pdf(filename):
    return send_from_directory("static/material/S2/Professional communication   HUN 102", filename)

#Vector Calculus & Differential Equations and Transforms  MAT102
@app.route("/material/S2/Vector Calculus & Differential Equations and Transforms  MAT102/s1")
def s2_mat102():
    return render_template('material/S2/Vector Calculus & Differential Equations and Transforms  MAT102/s1.html')

@app.route("/material/S2/Vector Calculus & Differential Equations and Transforms  MAT102/<filename>")
def s2_mat102_pdf(filename):
    return send_from_directory("static/material/S2/Vector Calculus & Differential Equations and Transforms  MAT102", filename)





#s3
@app.route("/material/S3/s3")
def s3():
    return render_template('material/S3/s3.html')

 
@app.route("/material/S3/Data structure/demo")
def s3_data_structures():
    return render_template('material/S3/Data structure/demo.html')

@app.route("/material/S3/Data structure/<filename>")
def s3_data_structures_pdf(filename):
    return send_from_directory("static/material/S3/Data structure", filename)

@app.route("/material/S3/Digital System Design/demo")
def s3_digital_system_design():
    return render_template('material/S3/Digital System Design/demo.html')


@app.route("/material/S3/Digital System Design/<filename>")
def s3_digital_system_design_pdf(filename):
    return send_from_directory("static/material/S3/Digital System Design", filename)


@app.route("/material/S3/Discrete Mathematics Structure/demo")
def s3_discrete_math():
    return render_template('material/S3/Discrete Mathematics Structure/demo.html')


@app.route("/material/S3/Discrete Mathematics Structure/<filename>")
def s3_discrete_math_pdf(filename):
    return send_from_directory("static/material/S3/Discrete Mathematics Structure", filename)

@app.route("/material/S3/Problem Solving using Python/demo")
def s3_python():
    return render_template('material/S3/Problem Solving using Python/demo.html')


@app.route("/material/S3/Problem Solving using Python/<filename>")
def s3_python_pdf(filename):
    return send_from_directory("static/material/S3/Problem Solving using Python", filename)


@app.route("/material/S3/professional ethics/demo")
def s3_professional_ethics():
    return render_template('material/S3/professional ethics/demo.html')

@app.route("/material/S3/professional ethics/<filename>")
def s3_professional_ethics_pdf(filename):
    return send_from_directory("static/material/S3/professional ethics", filename)

@app.route("/material/S3/sustainable/demo")
def s3_sustainable_development():
    return render_template('material/S3/sustainable/demo.html')

@app.route("/material/S3/sustainable/<filename>")
def s3_sustainable_dev_pdf(filename):
    return send_from_directory("static/material/S3/sustainable", filename)

@app.route("/material/S4/s4")
def s4():
    return render_template('material/S4/s4.html')


# Semester 4 routes
@app.route("/material/S4/oot/demo")
def s4_oot():
    return render_template('material/S4/oot/demo.html')

@app.route("/material/S4/oot/<filename>")
def s4_oot_pdf(filename):
    return send_from_directory("static/material/S4/oot", filename)

@app.route("/material/S4/dbms/demo")
def s4_dbms():
    return render_template('material/S4/dbms/demo.html')

@app.route("/material/S4/dbms/<filename>")
def s4_dbms_pdf(filename):
    return send_from_directory("static/material/S4/dbms", filename)

@app.route("/material/S4/Design and Engineering/demo")
def s4_design_engineering():
    return render_template('material/S4/Design and Engineering/demo.html')

@app.route("/material/S4/Design and Engineering/<filename>")
def s4_design_engineering_pdf(filename):
    return send_from_directory("static/material/S4/Design and Engineering", filename)

@app.route("/material/S5/s5")
def s5():
    return render_template('material/S5/s5.html')


# Semester 5 routes
@app.route("/material/S5/WAD/demo")
def s5_wad():
    return render_template('material/S5/WAD/demo.html')

@app.route('/material/S5/WAD/<filename>')
def s5_web_pdf(filename):
    return send_from_directory("static/material/S5/WAD", filename)

@app.route("/material/S5/OS/demo")
def s5_os():
    return render_template('material/S5/OS/demo.html')

@app.route('/material/S5/OS/<filename>')
def s5_os_pdf(filename):
    return send_from_directory("static/material/S5/OS", filename)

@app.route("/material/S5/DCN/demo")
def s5_dcn():
    return render_template('material/S5/DCN/demo.html')

@app.route("/material/S5/DCN/<filename>")
def s5_dcn_pdf(filename):
    return send_from_directory("static/material/S5/DCN", filename)


@app.route("/material/S5/FLAT/demo")
def s5_flat():
    return render_template('material/S5/FLAT/demo.html')

@app.route('/material/S5/FLAT/<filename>')
def s5_flat_pdf(filename):
    return send_from_directory("static/material/S5/FLAT", filename)

@app.route("/material/S5/MSE/demo")
def s5_mse():
    return render_template('material/S5/MSE/demo.html')


@app.route('/material/S5/MSE/<filename>')
def s5_mse_pdf(filename):
    return send_from_directory("static/material/S5/MSE", filename)

@app.route("/material/S5/Disaster/demo")
def s5_disaster():
    return render_template('material/S5/Disaster/demo.html')


@app.route("/material/S5/Disaster/<filename>")
def s5_disaster_management_pdf(filename):
    return send_from_directory("static/material/S5/Disaster", filename)




@app.route("/material/S6/s6")
def s6():
    return render_template('material/S6/s6.html')


# Semester 6 routes
@app.route("/material/S6/elective/demo")
def s6_elective():
    return render_template('material/S6/elective/demo.html')


@app.route('/material/S6/elective/<filename>')
def s6_elective_pdf(filename):
    return send_from_directory("static/material/S6/elective", filename)

@app.route("/material/S6/TCP IP/demo")
def s6_tcp_ip():
    return render_template('material/S6/TCP IP/demo.html')

@app.route('/material/S6/TCP IP/<filename>')
def s6_tcpip_pdf(filename):
    return send_from_directory("static/material/S6/TCP IP", filename)

@app.route("/material/S6/aad/demo")
def s6_aad():
    return render_template('material/S6/aad/demo.html')

@app.route('/material/S6/aad/<filename>')
def s6_aad_pdf(filename):
    return send_from_directory("static/material/S6/aad", filename)

@app.route("/material/S6/Data science/demo")
def s6_data_science():
    return render_template('material/S6/Data science/demo.html')

@app.route('/material/S6/Data science/<filename>')
def s6_ds_pdf(filename):
    return send_from_directory("static/material/S6/Data science", filename)

@app.route("/material/S6/IEFT/demo")
def s6_ieft():
    return render_template('material/S6/IEFT/demo.html')

@app.route('/material/S6/IEFT/<filename>')
def s6_ieft_pdf(filename):
    return send_from_directory("static/material/S6/IEFT", filename)

@app.route("/material/S6/ccw/demo")
def s6_ccw():
    return render_template('material/S6/ccw/demo.html')

@app.route('/material/S6/ccw/<filename>')
def s6_ccw_pdf(filename):
    return send_from_directory("static/material/S6/ccw", filename)


@app.route("/material/S7/s7")
def s7():
    return render_template('material/S7/s7.html')


# Semester 7 routes
@app.route("/material/S7/Data Analytics/demo")
def s7_data_analytics():
    return render_template('material/S7/Data Analytics/demo.html')

@app.route('/material/S7/Data Analytics/<filename>')
def s7_da_pdf(filename):
    return send_from_directory("static/material/S7/Data Analytics", filename)

@app.route("/material/S7/Open Elective/demo")
def s7_open_elective():
    return render_template('material/S7/Open Elective/demo.html')

@app.route("/material/S7/Industrial Safety Engineering/demo")
def s7_industrial_safety():
    return render_template('material/S7/Industrial Safety Engineering/demo.html')

@app.route('/material/S2/Industrial Safety Engineering/<filename>')
def s2_course_pdf(filename):
    return send_from_directory("static/material/S7/Industrial Safety Engineering", filename)


@app.route("/material/S7/Elective - II/demo")
def s7_elective_ii():
    return render_template('material/S7/Elective - II/demo.html')
@app.route('/material/S7/Elective - II/<filename>')
def s2_math_pdf(filename):
    return send_from_directory('static/material/S7/Elective - II', filename)


@app.route("/material/S8/s8")
def s8():
    return render_template('material/S8/s8.html')


# Semester 8 routes
@app.route("/material/S8/CNS/demo")
def s8_cns():
    return render_template('material/S8/CNS/demo.html')
@app.route('/material/S8/CNS/<filename>')
def s8_cns_pdf(filename):
    return send_from_directory("static/material/S8/CNS", filename)

@app.route("/material/S8/Comprehensive Viva/demo")
def s8_comprehensive_viva():
    return render_template('material/S8/Comprehensive Viva/demo.html')

@app.route("/material/S8/Elective - III/demo")
def s8_elective_iii():
    return render_template('material/S8/Elective - III/demo.html')

@app.route("/material/S8/Elective - IV/demo")
def s8_elective_iv():
    return render_template('material/S8/Elective - IV/demo.html')

@app.route("/material/S8/Elective - V/demo")
def s8_elective_v():
    return render_template('material/S8/Elective - V/demo.html')




# Semester 1 Question Paper Routes
@app.route("/previous_questions/S1")
def s1_question_papers():
    return render_template('previous_questions/S1/s1.html')
@app.route('/previous_questions/S1/CALCLUS/demo')
def s1_maths_questions():
    return render_template('previous_questions/S1/CALCLUS/demo.html')

@app.route("/previous_questions/S1/CALCLUS/<filename>")
def s1_mathsquestion_pdf(filename):
    return send_from_directory("static/previous_questions/S1/CALCLUS", filename)

@app.route('/previous_questions/S1/BEE & BEC/demo')
def s1_civil_mech_questions():
    return render_template('previous_questions/S1/BEE & BEC/demo.html')


@app.route("/previous_questions/S1/BEE & BEC/<filename>")
def s1_BEEBECquestion_pdf(filename):
    return send_from_directory("static/previous_questions/S1/BEE & BEC", filename)

@app.route('/previous_questions/S1/Eng Mechanics/demo')
def s1_lifeskills_questions():
    return render_template('previous_questions/S1/Eng Mechanics/demo.html')

@app.route("/previous_questions/S1/Eng Mechanics/<filename>")
def s1_EngMechanicsquestion_pdf(filename):
    return send_from_directory("static/previous_questions/S1/Eng Mechanics", filename)

@app.route('/previous_questions/S1/Graphics/demo')
def s1_graphics_questions():
    return render_template('previous_questions/S1/Graphics/demo.html')

@app.route("/previous_questions/S1/Graphics/<filename>")
def s1_Graphicsquestion_pdf(filename):
    return send_from_directory("static/previous_questions/S1/Graphics", filename)

@app.route('/previous_questions/S1/PHYSICS/demo')
def s1_physics_questions():
    return render_template('previous_questions/S1/PHYSICS/demo.html')


@app.route("/previous_questions/S1/PHYSICS/<filename>")
def s1_PHYSICSquestion_pdf(filename):
    return send_from_directory("static/previous_questions/S1/PHYSICS", filename)

# Semester 2 Question Paper Routes

@app.route("/previous_questions/S2")
def s2_question_papers():
    return render_template('previous_questions/S2/s2.html')

@app.route('/previous_questions/S2/BME & BCE/demo')
def s2_electrical_electronics():
    return render_template('previous_questions/S2/BME & BCE/demo.html')


@app.route('/previous_questions/S2/Chemistry/demo')
def s2_chemistry():
    return render_template('previous_questions/S2/Chemistry/demo.html')

@app.route('/previous_questions/S2/Professional communication/demo')
def s2_professional_comm():
    return render_template('previous_questions/S2/Professional communication/demo.html')


@app.route('/previous_questions/S2/Programming in c/demo')
def s2_c_programming():
    return render_template('previous_questions/S2/Programming in c/demo.html')


@app.route('/previous_questions/S2/Vector Calculus & Differential Equations and Transforms/demo')
def s2_vector_calculus():
    return render_template('previous_questions/S2/Vector Calculus & Differential Equations and Transforms/demo.html')



# Semester 3 Question Paper Routes
@app.route('/previous_questions/s3')
def s3_question_papers():
    return render_template('previous_questions/S3/s3.html')

@app.route('/previous_questions/S3/Data structures/demo')
def s3_data_structure():
    return render_template('previous_questions/S3/Data structures/demo.html')

@app.route('/previous_questions/S3/Digital system design/demo')
def s3_digital_systems():
    return render_template('previous_questions/S3/Digital system design/demo.html')

@app.route('/previous_questions/S3/Problem solving using python/demo')
def s3_python_paper():
    return render_template('previous_questions/S3/Problem solving using python/demo.html')

@app.route('/previous_questions/S3/Sustainable/demo')
def s3_sustainable_dev():
    return render_template('previous_questions/S3/Sustainable/demo.html')



# Semester 4 Question Paper Routes
@app.route('/previous_questions/s4')
def s4_question_papers():
    return render_template('previous_questions/S4/s4.html')  # Make sure this template exists


@app.route('/previous_questions/S4/Computer organization/demo')
def s4_computer_org():
    return render_template('previous_questions/S4/Computer organization/demo.html')

@app.route('/previous_questions/S4/Database management system/demo')
def s4_dbms_paper():
    return render_template('previous_questions/S4/Database management system/demo.html')

@app.route('/previous_questions/S4/Object oriented techniques (Java)/demo')
def s4_oop_java():
    return render_template('previous_questions/S4/Object oriented techniques (Java)/demo.html')


# Semester 5 Question Paper Routes
@app.route('/previous_questions/s5')
def s5_question_papers():
    return render_template('previous_questions/S5/s5.html')

@app.route('/previous_questions/S5/DCN/demo')
def s5_data_paper():
    return render_template('previous_questions/S5/DCN/demo.html')

@app.route('/previous_questions/S5/FLAT/demo')
def s5_automata_paper():
    return render_template('previous_questions/S5/FLAT/demo.html')

@app.route('/previous_questions/S5/MSE/demo')
def s5_disaster_paper():
    return render_template('previous_questions/S5/MSE/demo.html')

@app.route('/previous_questions/S5/OS/demo')
def s5_operating_systems():
    return render_template('previous_questions/S5/OS/demo.html')

@app.route('/previous_questions/S5/Software management/demo')
def s5_software_paper():
    return render_template('previous_questions/S5/Software management/demo.html')

@app.route('/previous_questions/S5/WAD/demo')
def s5_wad_papers():
    return render_template('previous_questions/S5/WAD/demo.html')



# Semester 6 Question Paper Routes

@app.route('/previous_questions/s6')
def s6_question_papers():
    return render_template('previous_questions/S6/s6.html')

@app.route('/previous_questions/S6/Algorithm analysis and desig/demo')
def s6_algorithm_design():
    return render_template('previous_questions/S6/Algorithm analysis and design/demo.html')


@app.route('/previous_questions/S6/Data science/demo')
def s5_wad_paper():
    return render_template('previous_questions/S6/Data science/demo.html')

@app.route('/previous_questions/S6/Elective/demo')
def s6_elective_paper():
    return render_template('previous_questions/S6/Elective/demo.html')

@app.route('/previous_questions/S6/Internetworking with TCP IP/demo')
def s6_tcpip():
    return render_template('previous_questions/S6/Internetworking with TCP IP/demo.html')

# 7
@app.route('/previous_questions/s7')
def s7_question_papers():
    return render_template('previous_questions/S7/s7.html')

@app.route('/previous_questions/S7/Data Analytics/demo')
def s7_dataanalytics():
    return render_template('previous_questions/S7/Data Analytics/demo.html')

@app.route('/previous_questions/S7/Machine learning/demo')
def s7_elective():
    return render_template('previous_questions/S7/Machine learning/demo.html')

@app.route('/previous_questions/S7/Mobile computing/demo')
def s7_mobile_computing():
    return render_template('previous_questions/S7/Mobile computing/demo.html')

@app.route('/previous_questions/S7/Open elective/demo')
def s7_open_elective_paper():
    return render_template('previous_questions/S7/Open elective/demo.html')


@app.route('/previous_questions/s8')
def s8_question_papers():
    return render_template('previous_questions/S8/s8.html')











@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        result = qa.invoke({"query": msg})
        return str(result["result"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)