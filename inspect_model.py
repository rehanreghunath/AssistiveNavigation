import tflite_runtime.interpreter as tflite
# Or try simple import if the environment has it
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path="app/src/main/assets/yolov8n_int8.tflite")
except:
    # If standard tensorflow is missing, use tflite_runtime
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path="app/src/main/assets/yolov8n_int8.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Inputs:")
for i in input_details:
    print(i['name'], i['shape'], i['dtype'])

print("Outputs:")
for o in output_details:
    print(o['name'], o['shape'], o['dtype'])
