"""小規模ユーティリティ関数群

リスト分割、ファイル列挙、CSV ヘッダ付与、補間、ブラー、フレーム時間変換など、
特定の機能に属さない汎用ヘルパを集めたモジュール。
"""
import os
import sys
import glob
import cv2
import numpy as np


def split_list_based_on_elements(original_list, k):
    total_elements = sum([len(sublist) for sublist in original_list])
    average_elements = total_elements / k
    # Splitting original_list
    split_lists = []
    current_list = []
    current_count = 0
    residue = 0  # 余りの要素数

    for sublist in original_list:
        if current_count + len(sublist) <= average_elements + residue:
            current_list.append(sublist)
            current_count += len(sublist)
        else:
            split_lists.append(current_list)
            residue += current_count - average_elements
            current_list = [sublist]
            current_count = len(sublist)
            if len(split_lists) == k - 1:
                split_lists.append(original_list[original_list.index(sublist):])
                return split_lists

    if current_list:
        split_lists.append(current_list)

    return split_lists

def split_array(arr, N):
    avg_len = len(arr) // N
    remain = len(arr) % N
    splits = []

    i = 0
    for _ in range(N):
        split_len = avg_len + (1 if remain > 0 else 0)
        splits.append(arr[i:i+split_len].tolist())
        i += split_len
        remain -= 1

    return splits

def split_uniq_based_on_original(uniq_list, split_original_list):
    split_uniq = []
    index = 0
    for split_list in split_original_list:
        split_len = len(split_list)
        if isinstance(uniq_list, np.ndarray):
            split_uniq.append(uniq_list[index:index+split_len].tolist())
        else:
            split_uniq.append(uniq_list[index:index+split_len])
        index += split_len
    return split_uniq

# パス内の映像ファイルを抽出して配列を返す
def addmovfile(prepath):
    file_list=glob.glob(prepath+"/*.MOV")
    file_list.extend(glob.glob(prepath+"/*.mp4"))
    file_list.extend(glob.glob(prepath+"/*.mov"))
    file_list.extend(glob.glob(prepath+"/*.MP4"))
    print(file_list)
    return file_list

# 映像ファイルのfpsを抽出して配列を返す
def returnfps(path_lists):
    results = []
    for path in path_lists:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)# fps
        results.append(fps)
    return results

def closest_value(start_value, end_value, num_values,zRange):
    log_start = np.log10(start_value)
    log_end = np.log10(end_value)
    rounded_values = []
    for i in range(num_values):
        log_value = np.interp(i, [0, num_values - 1], [log_start, log_end])
        value = int(round(10 ** log_value, -int(np.floor(log_value))))
        rounded_values.append(value)
    return min(rounded_values, key=lambda x: abs(x - zRange))
#blowupTransのTimePoint,timevalueの値からの調整
def search(num,frame_nums,timepoints):
    vvv=0
    for h in range(0,len(timepoints)-1):
        top=timepoints[h]*frame_nums
        bottom=timepoints[h+1]*frame_nums
        if num >= top and num < bottom:
            vvv=h
    return vvv

def addCsvHeader(d):
    print(sys._getframe().f_code.co_name)
    # 列と行のヘッダーを定義
    if len(d.shape)==2:
        column_header = [ i for i in range(d.shape[1]+1)]
        column_footer = [ i for i in range(d.shape[1]+1)]
        column_footer[1:5]=[np.min(d),np.max(d),np.max(d)-np.min(d),d.shape[0]]
        row_header = [i for i in range(d.shape[0])]
    else :
        column_header = [ i for i in range(d.shape[0]+1)]
        column_footer = [ i for i in range(d.shape[0]+1)]
        column_footer[1:5]=[np.min(d),np.max(d),np.max(d)-np.min(d),d.shape[0]]
        row_header = [0]
         
    # 列と行のヘッダーを追加した新しい配列を作成
    d=np.hstack((np.array(row_header).reshape(-1, 1), d))
    d = np.vstack((np.array(column_header), d))
    d = np.vstack((d,np.array(column_footer)))
    return d

#再生断面軌道の編集のログを出力。2023.10.19 added
def append_to_logfile(text_to_append):
        with open("maneuverlog.txt", "a") as file:
            file.write(text_to_append + "\n")

def bezier_interpolation(p0, p1, p2, t):
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

# ファイル名を数値部分でソートするための関数
def extract_number(filename):
    parts = filename.rsplit('_', 1)  # ファイル名を最後の'_'で分割
    number_part = parts[-1].split('.')[0]  # ナンバリング部分を取得 ('sep-1'のような形式)
    number = int(number_part.split('-')[-1])  # 数値の部分を抽出して整数に変換
    return number

# ファイル名からベース文字列を抽出する関数
def extract_base_string(file_name):
    parts = file_name.rsplit('_', 1)  # ファイル名を最後の'_'で分割
    return parts[0] + '_'  # ベースとなる文字列を返す

def convert_npy_to_jpg(npy_file_path, output_folder):
    # Numpy ファイルを読み込む
    data = np.load(npy_file_path)
    
    # 出力ファイル名を作成（拡張子を .jpg に変更）
    output_file_name = npy_file_path.split('/')[-1].replace('.npy', '.jpg')
    output_path = f"{output_folder}/{output_file_name}"
    
    # BGR 形式に変換して JPG として保存
    cv2.imwrite(output_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
    
    return output_path

# convert_npys_to_video は rendered_npys_to_mov(per_frame=True) に統合済み
# 使用例: rendered_npys_to_mov(out_dir, npys_path='frames/', per_frame=True, out_fps=30)


def custom_blur(data, s_frame, e_frame, bl_time, dim_num=1):
    time_array = data[:,:,dim_num]
    s_frame=np.clip(s_frame,0,time_array.shape[0])
    e_frame=np.clip(e_frame,0,time_array.shape[0])
    blur_array = np.zeros([e_frame - s_frame, data.shape[1]], dtype=np.float64)

    # ブラー時間が短い場合の調整
    if e_frame - s_frame < bl_time :
        bl_time = (e_frame - s_frame) 

    for y in range(e_frame - s_frame):
        # フレームの位置に応じてブラー適応範囲を動的に決定
        if y < bl_time / 2:
            # s_frameに近い場合は、範囲を縮小
            apply_bl_time = y
        elif (e_frame - s_frame) - y - 1 < bl_time / 2:
            # e_frameに近い場合は、範囲を縮小
            apply_bl_time = (e_frame - s_frame) - y - 1
        else:
            apply_bl_time = bl_time // 2

        start_index =int(max(s_frame + y - apply_bl_time, 0))
        end_index = int(min(s_frame + y + apply_bl_time + 1, time_array.shape[0]))
        blur_range = time_array[start_index:end_index, :]
        # print(y,apply_bl_time,start_index,end_index)
        blur_array[y, :] = np.mean(blur_range, axis=0)

    # 更新されたブラー配列を元のデータに代入
    data[s_frame:e_frame,:,dim_num] = blur_array
    return data


def custom_onedimention_blur(time_array, s_frame, e_frame, bl_time):
    s_frame=np.clip(s_frame,0,time_array.shape[0])
    e_frame=np.clip(e_frame,0,time_array.shape[0])
    
    blur_array = np.zeros([e_frame - s_frame, time_array.shape[1]], dtype=np.float64)

    # ブラー時間が短い場合の調整
    if e_frame - s_frame < bl_time :
        bl_time = (e_frame - s_frame) 

    for y in range(e_frame - s_frame):
        # フレームの位置に応じてブラー適応範囲を動的に決定
        if y < bl_time / 2:
            # s_frameに近い場合は、範囲を縮小
            apply_bl_time = y
        elif (e_frame - s_frame) - y - 1 < bl_time / 2:
            # e_frameに近い場合は、範囲を縮小
            apply_bl_time = (e_frame - s_frame) - y - 1
        else:
            apply_bl_time = bl_time // 2

        start_index =int(max(s_frame + y - apply_bl_time, 0))
        end_index = int(min(s_frame + y + apply_bl_time + 1, time_array.shape[0]))
        blur_range = time_array[start_index:end_index, :]
        # print(y,apply_bl_time,start_index,end_index)
        blur_array[y, :] = np.mean(blur_range, axis=0)

    # 更新されたブラー配列を元のデータに代入
    time_array[s_frame:e_frame] = blur_array
    return time_array

def onedimention_LoopBlur(time_array,blur):
    array = np.vstack((np.vstack((time_array,time_array)),time_array))
    if blur > 0 : array=cv2.blur(array,(1,int(blur)))
    return array[time_array.shape[0]:time_array.shape[0]+time_array.shape[0],:]

# フレームを時間に変換する関数
def frames_to_min_sec(frames, fps):
    seconds = int(frames // fps)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"  # 時:分:秒 形式
    elif minutes > 0:
        return f"{minutes}:{seconds:02d}"  # 分:秒 形式
    else:
        return f"{seconds}s"  # 秒のみ
    
#2024,10/21に追加
def calculate_parallel_perpendicular(A, B, q,previous_AB):
    # ベクトルABとAqを計算
    AB = B - A
    Aq = q - A
    
    # ベクトルABの長さを計算
    AB_length_squared = np.dot(AB, AB)
    
    if AB_length_squared == 0:
        # print("ABの長さが0")
        AB = previous_AB
        AB_length_squared=[1.0,1.0]
        # return 0, 0  # 長さが0なら、並行・垂直成分も0
    
    # 平行成分の計算（ベクトルAqをABに射影）
    parallel_component = (np.dot(AB, Aq) / AB_length_squared) * AB
    
    # 垂直成分の計算（Aqベクトルから平行成分を引いたもの）
    perpendicular_component = Aq - parallel_component
    # print(parallel_component,perpendicular_component)
    # ベクトルの大きさを計算
    # return np.linalg.norm(parallel_component),np.linalg.norm(perpendicular_component)
    parallel= parallel_component[1] if parallel_component[0] == 0 else parallel_component[0]
    perpendicular=perpendicular_component[1] if perpendicular_component[0] == 0 else perpendicular_component[0]
    return parallel,perpendicular

def create_video_from_images(image_folder, output_folder, output_filename, frame_rate=30):
    """
    指定されたディレクトリ内のPNGファイルから動画を作成する関数。

    Args:
        image_folder (str): PNG画像が保存されているディレクトリのパス。
        output_folder (str): 出力する動画ファイルを保存するディレクトリのパス。
        output_filename (str): 出力する動画ファイルの名前（拡張子を含む）。
        frame_rate (int, optional): 動画のフレームレート。デフォルトは30FPS。

    Returns:
        str: 作成された動画ファイルのフルパス。
    """

    # 画像ファイルのリストを取得し、ソートして順番を整える
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    if not images:
        raise ValueError("指定されたフォルダにPNGファイルが見つかりませんでした。")

    # 最初の画像を読み込んで、フレームサイズを取得
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_frame.shape

    # 出力ファイルのフルパスを作成
    output_path = os.path.join(output_folder, output_filename)

    # 動画ファイルを書き出すためのVideoWriterオブジェクトを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4形式のコーデック
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    total_images = len(images)

    # すべての画像を順番に読み込み、動画ファイルに書き込む
    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

        # 進捗の報告
        progress = (i + 1) / total_images * 100
        sys.stdout.write(f"\rvideo rendering: {progress:.2f}% ({i + 1}/{total_images})")
        sys.stdout.flush()

    # 作成した動画ファイルを閉じる
    self._close_video_sink(_sink_kind, _sink_obj)

    print(f"映像の作成が完了しました: {output_path}")
    return output_path

    
#2024,11/16 に追加
def double_first_dimension_with_interpolation(arr,next_first_array=None):
    """
    多次元配列の第一次元を2倍にし、要素間を線形補完する。

    Parameters:
        arr (np.ndarray): 元の多次元配列（shape=(n, ..., ...)）

    Returns:
        np.ndarray: 第一次元が2倍に拡張された配列
    """
    # 元の第一次元のサイズ
    n, *other_dims = arr.shape
    
    # 新しい第一次元のサイズ
    new_n = n * 2  if isinstance(next_first_array, np.ndarray) else n * 2 - 1
    
    # 補完結果を格納する配列
    expanded_array = np.zeros((new_n, *other_dims), dtype=arr.dtype)
    
    # 元のデータを新しい配列にコピー
    expanded_array[::2] = arr  # 元のデータは奇数位置にそのまま配置
    
    # 補完処理（線形補完）
    if isinstance(next_first_array, np.ndarray) :
        for i in range(1, new_n-1, 2):  # 偶数位置に補完
            expanded_array[i] = (arr[i // 2] + arr[i // 2 + 1]) / 2
        expanded_array[-1] = (arr[-1] + next_first_array) / 2
    else :
        for i in range(1, new_n, 2):  # 偶数位置に補完
            expanded_array[i] = (arr[i // 2] + arr[i // 2 + 1]) / 2
    return expanded_array


