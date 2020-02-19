import matplotlib.pyplot as plt

def setsettings(type="original"):
    if type == "original":

        plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
        plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
        plt.rcParams["font.size"] = 30  # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize'] = 25  # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize'] = 25  # 軸だけ変更されます
        plt.rcParams['xtick.direction'] = 'in'  # x axis in
        plt.rcParams['ytick.direction'] = 'in'  # y axis in
        plt.rcParams['axes.linewidth'] = 1.0  # axis line width
        plt.rcParams['axes.grid'] = True  # make grid
        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
        plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
        plt.rcParams["legend.labelspacing"] = 0.  # 垂直方向（縦）の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = 1.  # 凡例の線と文字の距離の長さ
        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale

    elif type == "seminer":
        #時間が無くてまだ作ってないけど、ゼミ発表などでは数値をもう少し大きくしないと見えない

        plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
        plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
        plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize'] = 9  # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize'] = 10  # 軸だけ変更されます
        plt.rcParams['xtick.direction'] = 'in'  # x axis in
        plt.rcParams['ytick.direction'] = 'in'  # y axis in
        plt.rcParams['axes.linewidth'] = 1.0  # axis line width
        plt.rcParams['axes.grid'] = True  # make grid
        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
        plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
        plt.rcParams["legend.labelspacing"] = 0.  # 垂直方向（縦）の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = 1.  # 凡例の線と文字の距離の長さ
        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale

    else:
        pass