1. 先在`yuxiu/simple-HRNet`项目下运行`scripts/generate_json.py`，参数`--videos_path`指的是一个装满同一class视频的文件夹，json将会存在该文件夹内。每个类别的文件夹都要运行一次以生成该类的json。
2. 将所有类的json拷贝到同一个文件夹中
3. 在`yuxiu/ST-GCN`项目下运行`tools/json_gendata.py`，参数`--data_path`就是装着所有json的文件夹，之后所有json会被装载到npk和pkl文件里，这就是`ST-GCN`训练的数据格式，存在`out_folder`里