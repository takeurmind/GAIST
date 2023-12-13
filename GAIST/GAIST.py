from flask import Flask, render_template, request, flash, url_for, redirect, send_from_directory
import os
import csv
import zipfile
import shutil
import modelcall
import time
from datetime import datetime
app = Flask(__name__)

csv_file_name = ''
UPLOAD_FOLDER = './GAIST/uploads'
CSV_FOLDER = './GAIST/surveillance'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['csv_file_name'] = csv_file_name


app.secret_key = 'gaistrocks'
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def read_csv_except_first_line(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            data.append(row)
    return data


@app.route('/')
def index():
    return render_template('index.html')

def read_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

@app.route('/team-members')
def team_members():
    return render_template('team-members.html')

@app.route('/class-test-results')
def class_test_results():
    classifying_result_path = app.config['csv_file_name']
    surveillance_data_path = './GAIST/surveillance/surveillance_data.csv'

    if os.path.exists(classifying_result_path):
        result_file_path = classifying_result_path
        mode_title = 'Classifying Mode'
    else:
        return render_template('loading.html')

    
    csv_data = read_csv(result_file_path)

    csv_data2_content = read_csv(surveillance_data_path)
    csv_data2 = csv_data2_content[-1]
    csv_data2 = csv_data2[0]
    csv_data2 = csv_data2.split("\t")
    print(csv_data2)
    if csv_data2[1] == 'cpuIpex':
        a = 'GAIST_optimized_CPU'
    else:
        a = csv_data2[1]

    tab_dict = {
        'mode' : a,
        'model' : csv_data2[2],
        'file' : csv_data2[3],
        'time' : csv_data2[4][0:6],
    }
    
    return render_template('classifying-test-results.html', csv_data=csv_data, csv_data2=tab_dict, mode_title=mode_title)

@app.route('/bench-test-results')
def bench_test_results():
    benchmark_result_path = app.config['csv_file_name']

    if os.path.exists(benchmark_result_path):
        result_file_path = benchmark_result_path
        mode_title = 'Benchmark Test Mode'
    else:
        return render_template('loading.html')

    csv_data = read_csv_except_first_line(result_file_path)
    
    return render_template('benchmark-test-results.html', csv_data=csv_data, mode_title=mode_title)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash({'file_name': '', 'message': 'No file part'}, 'error')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash({'file_name': '', 'message': 'No selected file'}, 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        create_folder(app.config['UPLOAD_FOLDER'])
        create_folder(app.config['CSV_FOLDER'])

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        file_path = './GAIST/static/images/'
        csv_path = './GAIST/results/'
        if 'zip' in filename :
            zip_file_path = './GAIST/uploads/' + file.filename
            extracted_dir = './GAIST/static/images/'

            shutil.rmtree(extracted_dir)
            os.makedirs("./GAIST/static/images/")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)

            for i, filename in enumerate(os.listdir(extracted_dir), start=1):
                old_file_path = os.path.join(extracted_dir, filename)
                new_file_path = os.path.join(extracted_dir, f'{i}.jpg')
                os.rename(old_file_path, new_file_path)

        flash({'file_name': file.filename, 'message': 'File Uploaded Successfully.'}, 'success')

        mode = request.form.get('mode', '')
        model = request.form.get('model', '')
        select_mode = mode
        select_model = model
        active_tab = request.form.get('active_tab', '')
        nows = datetime.today()
        tnow = nows.strftime('%m-%d %H:%M')
        times = ''
        if active_tab == 'classifying-tab' :
            stime = time.time()
            model = modelcall.call_model(mode, model)
            csv_path = csv_path + tnow + '_' + file.filename.split('.')[0] +'_'+ select_mode +'_'+  select_model + '_result_data.csv'
           
            modelcall.process_images_from_zip(file_path,model, mode, csv_path)
            etime = time.time()
            times = etime-stime
        elif active_tab == 'benchmark-tab' :

            b_csv_path = './GAIST/results_bench/'
            mode = ['cpu', 'cpuIpex', 'gpu']
            model = ['mobilenet','resnet18', 'vgg16']

            x = datetime.now()
            b_csv_path = b_csv_path + x.strftime("%m월 %d일 %H시 %M분") + '_' +'bechmark_data.csv'
            csv_filename = os.path.join(b_csv_path)
            mode_i = ''
            model_i = ''
            for mode_index in mode:
                for model_index in model:
                    csv_paths = csv_path + 'bench_' + mode_index+ '_' + model_index+'.csv'

                    stime = time.time()
                    model_s = modelcall.call_model(mode_index, model_index)
                    modelcall.process_images_from_zip(file_path,model_s, mode_index, csv_paths)
                    etime = time.time()
                    times = etime-stime
                    
                   
                    if not os.path.exists(csv_filename):
                        with open(csv_filename, 'w', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile, delimiter=',')
                            csv_writer.writerow(['PROCESSOR MODE' , 'MODEL', 'INFERENCE TIME'  ])

                    with open(csv_filename, 'a', newline='') as csvfile:
                        if mode_index == 'cpuIpex':
                            mode_i = "GAIST"
                        elif mode_index == 'cpu': 
                            mode_i = "CPU"
                        elif mode_index == 'gpu': 
                            mode_i = "GPU"

                        if model_index == 'mobilenet': 
                            model_i = "MobileNet"
                        elif model_index == 'resnet18': 
                            model_i = "ResNet18"
                        elif model_index == 'vgg16': 
                            model_i = "VGG16"
                        csv_writer = csv.writer(csvfile, delimiter='\t')
                        csv_writer.writerow([ mode_i, model_i, f"{times:.2f}"])
            app.config['csv_file_name'] = b_csv_path
                            
        csv_filename = os.path.join(app.config['CSV_FOLDER'], 'surveillance_data.csv')

        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t')
                csv_writer.writerow(['Tab Content', 'Select Mode', 'Select Model', 'File Name', 'Elapsed Times' ])

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([active_tab, select_mode, select_model, file.filename , times])
        
            if active_tab == "classifying-tab":
                app.config['csv_file_name'] = csv_path
                return redirect(url_for('class_test_results'))
            else:
                return redirect(url_for('bench_test_results'))
    
    else:
        flash({'file_name': '', 'message': 'Invalid file type. Allowed types are png, jpg, jpeg, gif.'}, 'error')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)