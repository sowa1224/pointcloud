#ノート
# 白い点群データを抽出
#にんじんの点群を抽出するプログラムの順番：
#ply_to_csv→csv_hsv_extract→csv_RGB_extrac→csv_show_open3d_matlab
#点群ファイル　plyファイルを得る手法として　realsense -sdkによる手動　もしくは/home/sowa/prog/librealsense-2.54.1/wrappers/python/examples　でのexport...example2.plyのファイルを実行すると点群を得られる
#csvからplyファイルに変換するプログラムはcsv_to_ply_ver2.pyに残す
#年のためにplyファイルのままでhsv,半径外れ値処理を行ってplyファイルで保存する用のプログラムを作った　ファイルは　..._by_ply.pyになる


""""
#11/28
・点群データ処理
    ply_to_csv : plyファイルをx,y,z,r,g,b,nx,ny,nz,h,s,vの内容の点群データのcsvファイルを作成
    csv_hsv_exytact_complete_ver2: csvファイル内の環境点群から対象物（ジャガイモ）の点群を抽出,complete_ver2では一つのディレクトリ内のすべての連番のcsvファイルを一斉に処理が可能.また特定の一つのcsvファイルの場合はcompleteで
・ジャガイモの芽

・皮むき軌道＆ポリゴン



#1017
#じゃがいもの種類によって芽の場所を特定するのは困難　（hsvを頼りにやるには）　結果は

#2023/9/12
#下のコードは移動平均プログラムを表示と近似曲線を描く用のプログラム
#現在まだ移動平均ができているかどうかの確認をしている、できたら下を進む

""""