#include <bits/stdc++.h>
#define resolution 128
using namespace std;

namespace matrix {
//整个矩阵类
    class Matrix {
    public:
        vector<vector<double>> mat;

        int row, col;

        void Init (int SizeH,int SizeW,double x) {
            mat.resize(0);
            for (int i = 0; i < SizeH; ++i) {
                vector<double> tmp;
                for (int j = 0; j < SizeW;++j) {
                    tmp.push_back(x);
                }
                mat.push_back(tmp);
            }
            row = SizeH;
            col = SizeW;
        }

        void Print () {
            cout << "The Size of Matrix is : " << row << " * " << col << endl;

            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    if (mat[i][j]<0)
                        printf ("%.2lf ",mat[i][j]);
                    else
                        printf("%.3lf ", mat[i][j]);
                }
                cout << endl;
            }
        }

        void Photo () {
            cout << "The Size of Photo is : " << row << " * " << col << endl;

            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    if (abs (mat[i][j])<0.0001)
                        printf("0");
                    else
                        printf("1");
                }
                cout << endl;
            }

        }

        void Output () {
            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    printf ("%lf ",mat[i][j]);
                }
                cout << endl;
            }
        }

        double sum () {
            double res = 0;
            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    res += mat[i][j];
                }
            }
            return res;
        }

        Matrix operator + (const Matrix& m) {

            Matrix tmp;

            if (row!=m.row||col!=m.col)
                return tmp;

            tmp.Init(row, col, 0);

            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    tmp.mat[i][j] = mat[i][j] + m.mat[i][j];
                }
            }
            return tmp;
        }

        Matrix operator - (const Matrix& m) {

            Matrix tmp;

            if (row!=m.row||col!=m.col)
                return tmp;

            tmp.Init(row, col, 0);

            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    tmp.mat[i][j] = mat[i][j] - m.mat[i][j];
                }
            }
            return tmp;
        }


        Matrix operator * (const double& b) {

            Matrix tmp;

            tmp.Init(row, col, 0);

            for (int i = 0; i < row;++i) {
                for (int j = 0; j < col;++j) {
                    tmp.mat[i][j] = mat[i][j] * b;
                }
            }

            return tmp;
        }

    };



    //填充函数,将矩阵x上下各自填充r行,左右各自填充c列,填充数为num
    Matrix Full (Matrix x,int r,int c,double num) {
        int InputH = x.row;
        int InputW = x.col;

        int OutputH = InputH + r * 2;
        int OutputW = InputW + c * 2;

        Matrix tmp;
        tmp.Init(OutputH, OutputW, 0);


        //中间填充
        for (int i = r; i < r + InputH;++i) {
            for (int j = c; j < c + InputW;++j) {
                tmp.mat[i][j] = x.mat[i - r][j - c];
            }
        }

        //上下填充
        for (int i = 0; i < r;++i) {
            for (int j = 0; j < OutputW;++j) {
                tmp.mat[i][j] = tmp.mat[i + r + InputH][j] = num;
            }
        }

        //左右填充
        for (int i = 0; i < c; ++i) {
            for (int j = 0; j < OutputH;++j) {
                tmp.mat[j][i] = tmp.mat[j][i + c + InputW] = num;
            }
        }

        return tmp;
    }

    //卷积函数
    Matrix Cov (Matrix x,Matrix y) { //卷积核为y
        int InputH = x.row;
        int InputW = x.col;

        int covH = y.row;
        int covW = y.col;

        int OutputH = InputH - covH + 1;
        int OutputW = InputW - covW + 1;

        Matrix tmp;
        tmp.Init(OutputH, OutputW, 0);
        for (int i = 0; i < OutputH;++i) {
            for (int j = 0; j < OutputW;++j) {
                for (int r = 0; r < covH;++r) {
                    for (int c = 0; c < covW;++c) {
                        tmp.mat[i][j] += x.mat[i + r][j + c] * y.mat[r][c];
                    }
                }
            }
        }
        return tmp;
    }

    Matrix Flip (Matrix x) {
        int r = x.row;
        int c = x.col;
        Matrix tmp;
        tmp.Init(r, c, 0);

        for (int i = 0; i < r;++i) {
            for (int j = 0; j < c;++j) {
                tmp.mat[i][j] = x.mat[r - i - 1][c - j - 1];
            }
        }
        return tmp;
    }

    double Mul (Matrix a,vector<double> b) { //向量点乘
        double sum = 0;
        for (int i = 0; i < a.col;++i) {
            sum += a.mat[0][i] * b[i];
        }
        return sum;
    }

}
using namespace matrix;

struct convolutional_layer { //卷积层结构体

    int InputH; //输入图像高
    int InputW; //输入图像宽

    int Size;   //卷积核尺寸

    int InChannels; //输入通道数
    int OutChannels; //输出通道数

    vector<vector<Matrix>> Data;  //卷积核 总共 InChannels*OutChannels 个, 每个大小为 Size

    Matrix basicData; //偏置 个数为 OutChannels

    vector<Matrix> init;   //v 激活函数输入
    vector<Matrix> outit;  //y 激活函数输出
    vector<Matrix> d;      //局部梯度
};


struct pooling_layer {   //池化层结构体

    int InputW;   //输入图像宽
    int InputH;   //输入图像长

    int Size;     //池化窗口大小

    int InChannels; //输入通道数
    int OutChannels; //输出通道数

    int poolType;     //池化方法

    Matrix basicData; //偏置

    vector<Matrix> outit;         //y 激活函数输出
    vector<Matrix> d;            //局部梯度
    vector<Matrix> max_pos;      //最大池最大值位置
};

struct Output { //输出层结构体
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目

    Matrix Data;            //权值 大小为 inputNum * outputNum
    Matrix basicData; //偏置 个数为 outputNum

    Matrix init;     //v 激活函数输入
    Matrix outit;     //y 激活函数输出
    Matrix d;     //局部梯度
};

typedef convolutional_layer col_layer;
typedef pooling_layer pol_layer;
typedef Matrix Mat;

struct CNN {
    int LayerNum;

    col_layer C1;
    pol_layer S2;
    col_layer C3;
    pol_layer S4;
    col_layer C5;
    pol_layer S6;
    Output O7;

    Mat e;
    Mat L;
};

double Learn;

ifstream fin;

bool New_Train= true;

string Dataurl = "C:\\Users\\64783\\CLionProjects\\untitled\\cmake-build-debug\\NetWorkData.txt";

//卷积层结构体初始化
col_layer InitCovL (int InputH,int InputW,int Size,int InChannels,int OutChannels) {

    col_layer covl;

    if (!New_Train) {

        fin >> covl.InputH >> covl.InputW >> covl.Size >> covl.InChannels >> covl.OutChannels;

        for (int i = 0; i < covl.InChannels;++i) {
            vector<Mat> tmp;
            for (int j = 0; j < covl.OutChannels;++j) {
                Mat Tmp_;
                Tmp_.Init(covl.Size, covl.Size, 0); //初始化
                for (int r = 0; r < covl.Size;++r) {
                    for (int c = 0; c < covl.Size;++c) {
                        fin >> Tmp_.mat[r][c];
                    }
                }
                tmp.push_back(Tmp_);
            }
            covl.Data.push_back(tmp);
        }

        covl.basicData.Init(1, covl.OutChannels, 0);

        for (int i = 0; i < covl.OutChannels;++i)
            fin >> covl.basicData.mat[0][i];

        int OutH = covl.InputH;
        int OutW = covl.InputW;

        Mat tmp;

        tmp.Init(OutH, OutW, 0);

        for (int i = 0; i < OutChannels;++i) {
            covl.d.push_back (tmp);
            covl.init.push_back(tmp);
            covl.outit.push_back(tmp);
        }
    }
    else {
        // 下面是不读入文件形式的

        covl.InputH = InputH;
        covl.InputW = InputW;
        covl.Size = Size;

        covl.InChannels = InChannels;
        covl.OutChannels = OutChannels;

        for (int i = 0; i < InChannels;++i) {
            vector<Mat> tmp;
            for (int j = 0; j < OutChannels;++j) {
                Mat Tmp_;
                Tmp_.Init(Size, Size, 0); //初始化
                for (int r = 0; r < Size;++r) {
                    for (int c = 0; c < Size;++c) {
                        double Rd = (((double)rand() / (double)RAND_MAX) - 0.5) * 2;
                        Tmp_.mat[r][c] = Rd * sqrt(6.0 / (Size * Size * (InChannels + OutChannels)));
                    }
                }
                tmp.push_back(Tmp_);
            }
            covl.Data.push_back(tmp);
        }

        covl.basicData.Init(1, OutChannels, 0);

        int OutH = InputH;
        int OutW = InputW;

        Mat tmp;
        tmp.Init(OutH, OutW, 0);

        for (int i = 0; i < OutChannels;++i) {
            covl.d.push_back (tmp);
            covl.init.push_back(tmp);
            covl.outit.push_back(tmp);
        }
    }

    return covl;
}


pol_layer InitPolL(int InputH, int InputW, int Size, int InChannels, int OutChannels, int poolType) {
    pol_layer pool;
    if (!New_Train) {
        fin >> pool.InputH >> pool.InputW >> pool.Size >> pool.InChannels >> pool.OutChannels >> pool.poolType;

        pool.basicData.Init(1, pool.OutChannels, 0);

        int OutH = pool.InputH / Size;
        int OutW = pool.InputW / Size;

        Mat tmp;
        tmp.Init(OutH, OutW, 0);
        for (int i = 0; i < pool.OutChannels;++i) {
            pool.d.push_back (tmp);
            pool.outit.push_back(tmp);
            pool.max_pos.push_back(tmp);
        }
    }
    else {
        // 下面是不读入文件形式的

        pool.InputH = InputH;
        pool.InputW = InputW;
        pool.Size = Size;

        pool.InChannels = InChannels;
        pool.OutChannels = OutChannels;

        pool.poolType = poolType;


        pool.basicData.Init(1, OutChannels, 0);

        int OutH = InputH / Size;
        int OutW = InputW / Size;

        Mat tmp;
        tmp.Init(OutH, OutW, 0);
        for (int i = 0; i < OutChannels;++i) {
            pool.d.push_back (tmp);
            pool.outit.push_back(tmp);
            pool.max_pos.push_back(tmp);
        }

    }
    return pool;
}


Output initOutLayer(int inputNum, int outputNum) {
    Output outl;
    if (!New_Train) {

        fin >> outl.inputNum >> outl.outputNum;

        outl.basicData.Init(1, outl.outputNum, 0);
        outl.init.Init(1, outl.outputNum, 0);
        outl.outit.Init(1, outl.outputNum, 0);
        outl.d.Init(1, outl.outputNum, 0);

        for (int i = 0; i < outl.outputNum;++i)
            fin >> outl.basicData.mat[0][i];

        outl.Data.Init(outl.outputNum, outl.inputNum, 0);

        for(int i = 0; i < outl.outputNum; i++) {
            for(int j = 0; j < outl.inputNum; j++) {
                fin >> outl.Data.mat[i][j];
            }
        }

    }
    else {
        // 下面是不读入文件形式的

        outl.inputNum = inputNum;
        outl.outputNum = outputNum;

        outl.basicData.Init(1, outputNum, 0);
        outl.init.Init(1, outputNum, 0);
        outl.outit.Init(1, outputNum, 0);
        outl.d.Init(1, outputNum, 0);

        outl.Data.Init(outputNum, inputNum, 0);

        for(int i = 0; i < outputNum; i++) {
            for(int j = 0; j < inputNum; j++) {
                double Rd = (((double)rand() / (double)RAND_MAX) - 0.5) * 2;
                outl.Data.mat[i][j] = Rd * (sqrt(6.0 / (inputNum + outputNum)));
            }
        }
    }

    return outl;
}


#define full 0
#define same 1
#define valid 2

#define avgpol 0
#define maxpol 1

void Init (CNN &cnn,int InputH,int InputW,int InputNum,int OutputNum) { //初始化

    puts("Loading...");

    fin.open(Dataurl);
    cnn.LayerNum = 5;

    int covSize = 5;

    int polSize = 2;

    cnn.C1 = InitCovL(InputH, InputW, covSize, InputNum, 6); //第一层

    // 32-1+1=32
    InputH = InputH;
    InputW = InputW;

    cnn.S2 = InitPolL(InputH, InputW, polSize, cnn.C1.OutChannels, cnn.C1.OutChannels, maxpol); //第二层
    //32/2=16
    InputH /= polSize;
    InputW /= polSize;

    cnn.C3 = InitCovL(InputH, InputW, covSize, cnn.S2.OutChannels, 12); //第三层
    //16-5+1=12
    InputH = InputH;
    InputW = InputW;

    cnn.S4 = InitPolL(InputH, InputW, polSize, cnn.C3.OutChannels, cnn.C3.OutChannels, maxpol); //第四层
    //12/2=6
    InputH /= polSize;
    InputW /= polSize;

    cnn.C5 = InitCovL(InputH, InputW, covSize, cnn.S4.OutChannels, 24); //第三层
    //16-5+1=12
    InputH = InputH;
    InputW = InputW;

    cnn.S6 = InitPolL(InputH, InputW, polSize, cnn.C5.OutChannels, cnn.C5.OutChannels, maxpol); //第四层
    //12/2=6
    InputH /= polSize;
    InputW /= polSize;

    cnn.O7 = initOutLayer(InputH * InputW * cnn.S6.OutChannels, OutputNum);

    cnn.e.Init(1, OutputNum, 0);

    fin.close();

    puts("Loaded");

    return;
}

Mat correlation (Mat x,Mat CovData,int type) { //卷积函数 x对CovData卷积

    if (type==full) {    //Full 模式 对大小为(r,c)的卷积核卷积，原图像上下填充r-1,左右填充c-1,输出n+r-1行m+c-1列卷积结果
        return Cov(Full(x, CovData.row - 1, CovData.col - 1, 0), CovData);
    }
    else if (type==same) {    //Same 模式 对大小为(r,c)的卷积核卷积，原图像上下填充(r-1)/2,左右填充(c-1)/2,输出n行m列卷积结果
        return Cov(Full(x, (CovData.row - 1) / 2, (CovData.col - 1) / 2, 0), CovData);
    }
    else {    //Valid 模式 对大小为(r,c)的卷积核卷积，原图不填充，输出n-r+1行m-c+1列卷积结果
        return Cov(x, CovData);
    }

}

void avgPooling(Mat input, Mat &output, int Size) { //均值池化
    int OutH = input.row/Size;
    int OutW = input.col/Size;

    for (int i = 0; i < OutH;++i) {
        for (int j = 0; j < OutW;++j) {
            double sum = 0;
            for (int r = i * Size; r < i * Size + Size;++r) {
                for (int c = j * Size; c < j * Size + Size;++c) {
                    sum += input.mat[r][c];
                }
            }
            output.mat[i][j] = sum / ((double)(Size * Size));
        }
    }

    return;
}


void maxPooling(Mat input, Mat &output_pos, Mat &output, int Size) { //最大池化
    int OutH = input.row/Size;
    int OutW = input.col/Size;

    for (int i = 0; i < OutH;++i) {
        for (int j = 0; j < OutW;++j) {
            double max = -9999999.0;
            int max_pos = 0;
            for (int r = i * Size; r < i * Size + Size;++r) {
                for (int c = j * Size; c < j * Size + Size;++c) {
                    if (max<input.mat[r][c]) {
                        max = input.mat[r][c];
                        max_pos = r * input.col + c;
                    }
                }
            }
            output.mat[i][j] = max;
            output_pos.mat[i][j] = max_pos;
        }
    }

    return;
}

double relu (double x) {
    // return (1.0) / (1.0 - exp(-x));
    return fmax(x, 0.0);
}

double Der_relu (double x) {
    // return x * (1 - x);
    if (x>0)
        return 1;
    else
        return 0;
}

void softmax (Output &Out) {
    double sum = 0;

    int Outnum = Out.outputNum;

    for (int i = 0; i < Outnum;++i) {

        Out.outit.mat[0][i] = exp(Out.init.mat[0][i] + Out.basicData.mat[0][i]);
        sum += Out.outit.mat[0][i];

    }

    for (int i = 0; i < Outnum;++i) {

        Out.outit.mat[0][i] = Out.outit.mat[0][i] / sum;
    }

    return;
}


void Into_Col(vector<Mat> Input, int Type /* 卷积方式 */, col_layer &C) { //传入到卷积层


    for (int i = 0; i < C.OutChannels; ++i)
    {
        for (int j = 0; j < C.InChannels;++j) {
            Mat Out = correlation(Input[j], C.Data[j][i], Type);
            C.init[i] = C.init[i] + Out;
        }
        for (int r = 0; r < C.outit[i].row;++r) {
            for (int c = 0; c < C.outit[i].col;++c) {
                C.outit[i].mat[r][c] = relu(C.init[i].mat[r][c] + C.basicData.mat[0][i]);
            }
        }
    }

    return;
}

void Into_Pool(vector<Mat> Input, int Type /* 池化方式 */, pol_layer &S) { //传入到池化层

    if (Type==avgpol) { //均值池化

        for (int i = 0; i < S.OutChannels;++i) {
            avgPooling(Input[i], S.outit[i], S.Size);
        }

    }
    else { //最大池化

        for (int i = 0; i < S.OutChannels;++i) {
            maxPooling(Input[i], S.max_pos[i], S.outit[i], S.Size);
        }

    }

    return;

}

void Into_Out(vector<Mat> Input, Output &Out) { //传入到输出层

    Mat tmp;

    tmp.Init(1, Out.inputNum, 0);

    int Len = Input.size();

    int Row = Input[0].row;
    int Col = Input[0].col;

    for (int i = 0; i < Len;++i) {

        for (int r = 0; r < Row;++r) {

            for (int c = 0; c < Col;++c) {
                tmp.mat[0][i * Row * Col + r * Col + c] = Input[i].mat[r][c];
            }
        }
    }
    for (int i = 0; i < Out.init.col;++i) {
        Out.init.mat[0][i] = Mul(tmp, Out.Data.mat[i]);
    }

    softmax(Out);
}

void work (CNN &cnn,vector<Mat> Input) {

    Into_Col(Input, same, cnn.C1);

    Into_Pool(cnn.C1.outit, maxpol, cnn.S2);

    Into_Col(cnn.S2.outit, same, cnn.C3);

    Into_Pool(cnn.C3.outit, maxpol, cnn.S4);

    Into_Col(cnn.S4.outit, same, cnn.C5);

    Into_Pool(cnn.C5.outit, maxpol, cnn.S6);

    Into_Out(cnn.S6.outit, cnn.O7);


    return;
}

void Back_Softmax (Mat output,Mat &e,Output &Out) {
    for (int i = 0; i < Out.outputNum;++i) {
        e.mat[0][i] = Out.outit.mat[0][i] - output.mat[0][i];
    }


    for (int i = 0; i < Out.outputNum;++i) {
        Out.d.mat[0][i] = e.mat[0][i];
    }


    return;
}

void Back_Full (Output Out,pol_layer &S) {
    int OutH = S.InputH / S.Size;
    int OutW = S.InputW / S.Size;


    for (int i = 0; i < S.OutChannels;++i) {
        for (int r = 0; r < OutH;++r) {
            for (int c = 0; c < OutW;++c) {
                int w = i * OutH * OutW + r * OutW + c;

                for (int j = 0; j < Out.outputNum;++j) {
                    S.d[i].mat[r][c] = S.d[i].mat[r][c] + Out.d.mat[0][j] * Out.Data.mat[j][w];
                }
            }
        }
    }
    return;
}

Mat PushUp_avg (Mat M,int Size) {
    int NowR = M.row;
    int NowC = M.col;

    Mat res;

    res.Init(NowR * Size, NowC * Size, 0);

    for (int i = 0; i < NowR * Size;i+=Size) {

        for (int j = 0; j < NowC * Size;j+=Size) {

            for (int k = 0; k < Size;++k) {

                res.mat[i + k][j] = res.mat[i][j + k] = M.mat[i / Size][j / Size] / (Size * Size);
            }
        }
    }

    return res;
}

Mat PushUp_max (Mat M,Mat max_pos,int Size) {

    int NowR = M.row;
    int NowC = M.col;

    Mat res;

    res.Init(NowR * Size, NowC * Size, 0);


    for (int j = 0; j < NowR;++j) {

        for (int i = 0; i < NowC;++i) {
            int r = (int)(max_pos.mat[j][i]) / (NowC * Size);
            int c = (int)(max_pos.mat[j][i]) % (NowC * Size);

            res.mat[r][c] = M.mat[j][i];
        }
    }

    return res;
}

void Back_Pool (pol_layer S,col_layer &C) {

    for (int i = 0; i < C.OutChannels;++i) {
        Mat res;
        if (S.poolType==avgpol) {
            res = PushUp_avg(S.d[i], S.Size);
        }
        else {
            res = PushUp_max(S.d[i], S.max_pos[i], S.Size);
        }

        for (int r = 0; r < S.InputH;++r) {
            for (int c = 0; c < S.InputW;++c) {
                C.d[i].mat[r][c] = res.mat[r][c] * Der_relu(C.outit[i].mat[r][c]);
            }
        }
    }
}


void Back_Cov (col_layer C,int Type,pol_layer &S) {
    for (int i = 0; i < S.OutChannels;++i) {
        for (int j = 0; j < S.InChannels;++j) {
            Mat tmp;
            tmp = Flip(C.Data[i][j]);
            tmp = correlation(C.d[j], tmp, Type);
            S.d[i] = S.d[i] + tmp;
        }
    }
}

void Back (CNN &cnn,Mat Out) {
    Back_Softmax(Out, cnn.e, cnn.O7);
    Back_Full(cnn.O7, cnn.S6);
    Back_Pool(cnn.S6, cnn.C5);
    Back_Cov(cnn.C5, same, cnn.S4);
    Back_Pool(cnn.S4, cnn.C3);
    Back_Cov(cnn.C3, same, cnn.S2);
    Back_Pool(cnn.S2, cnn.C1);
    return;
}

void update_Full (vector <Mat> Input,Output &Out) {
    int OutR = Input[0].row;
    int OutC = Input[0].col;

    Mat tmp;

    tmp.Init(1, OutR * OutC * Input.size(), 0);

    for (int i = 0; i < Input.size();++i) {
        for (int r = 0; r < OutR;++r) {
            for (int c = 0; c < OutC;++c) {
                int w = i * OutR * OutC + r * OutC + c;
                tmp.mat[0][w] = Input[i].mat[r][c];
            }
        }
    }

    for (int j = 0; j < Out.outputNum;++j) {

        for (int i = 0; i < Out.inputNum;++i) {

            Out.Data.mat[j][i] = Out.Data.mat[j][i] - Learn * Out.d.mat[0][j] * tmp.mat[0][i];
        }

        Out.basicData.mat[0][j] = Out.basicData.mat[0][j] - Learn * Out.d.mat[0][j];
    }
}

void update_Cov (vector <Mat> Input,col_layer &Out) {
    for (int i = 0; i < Out.OutChannels;++i) {
        double sum = 0;
        for (int j = 0; j < Out.InChannels;++j) {
            Mat tmp;
            tmp = correlation(Full (Input[j],2,2,0), Out.d[i], valid); //这里要改下
            tmp = tmp * Learn;
            Out.Data[j][i] = Out.Data[j][i] - tmp;
        }
        sum = Out.d[i].sum ();
        Out.basicData.mat[0][i] -= sum * Learn;
    }
}

void Update (CNN &cnn,vector<Mat> Input) {

    update_Cov(Input, cnn.C1);

    update_Cov(cnn.S2.outit, cnn.C3);

    update_Cov(cnn.S4.outit, cnn.C5);

    update_Full(cnn.S6.outit, cnn.O7);

    return;
}

void clear_full (Output &Out) {
    for (int i = 0; i < Out.outputNum;++i) {
        Out.d.mat[0][i] = 0;
        Out.init.mat[0][i] = 0;
        Out.outit.mat[0][i] = 0;
    }
}

void clear_pol (pol_layer &Out) {
    for (int i = 0; i < Out.OutChannels;++i) {
        for (int r = 0; r < Out.d[0].row;++r) {
            for (int c = 0; c < Out.d[0].col;++c) {
                Out.d[i].mat[r][c] = 0;
                Out.outit[i].mat[r][c] = 0;
            }
        }
    }
}

void clear_cov (col_layer &Out) {
    for (int i = 0; i < Out.OutChannels;++i) {
        for (int r = 0; r < Out.d[0].row;++r) {
            for (int c = 0; c < Out.d[0].col;++c) {
                Out.d[i].mat[r][c] = 0;
                Out.init[i].mat[r][c] = 0;
                Out.outit[i].mat[r][c] = 0;
            }
        }
    }
}

void Clear (CNN &cnn) {
    clear_cov(cnn.C1);
    clear_pol(cnn.S2);
    clear_cov(cnn.C3);
    clear_pol(cnn.S4);
    clear_cov(cnn.C5);
    clear_pol(cnn.S6);
    clear_full(cnn.O7);
    return;
}


int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

//50000*3*32*32
/*
void read(string File,vector<vector<Mat>> &list,vector<Mat> &label) {
    ifstream file(File, ios::binary);
    if (file.is_open()) {
        for (int j = 0; j < 10000;++j) {
            unsigned char number_of_images = 0;
            file.read((char *) &number_of_images, sizeof(number_of_images));
            Mat tmp_;
            tmp_.Init(1, 10, 0);
            tmp_.mat[0][(int)(number_of_images)] = 1;
            label.push_back(tmp_);
            vector<Mat> photo;
            for (int num = 0; num < 3;++num) {
                Mat photo_tmp;
                photo_tmp.Init(32, 32, 0);
                for (int r = 0; r < 32;++r) {
                    for (int c = 0; c < 32;++c) {
                        unsigned char awa = 0;
                        file.read((char *)&awa, sizeof(awa));

                        photo_tmp.mat[r][c] = (double)(awa) / 255.0;
                    }
                }
                photo.push_back(photo_tmp);
            }
            list.push_back(photo);
        }
    }
    return;
}
 */


string ID[70] = {"apple",
                 "bat",
                 "beetle",
                 "bell",
                 "bird",
                 "Bone",
                 "bottle",
                 "brick",
                 "butterfly",
                 "camel",
                 "car",
                 "carriage",
                 "cattle",
                 "cellular_phone",
                 "chicken",
                 "children",
                 "chopper",
                 "classic",
                 "Comma",
                 "crown",
                 "cup",
                 "deer",
                 "device0",
                 "device1",
                 "device2",
                 "device3",
                 "device4",
                 "device5",
                 "device6",
                 "device7",
                 "device8",
                 "device9",
                 "dog",
                 "elephant",
                 "face",
                 "fish",
                 "flatfish",
                 "fly",
                 "fork",
                 "fountain",
                 "frog",
                 "Glas",
                 "guitar",
                 "hammer",
                 "hat",
                 "HCircle",
                 "Heart",
                 "horse",
                 "horseshoe",
                 "jar",
                 "key",
                 "lizzard",
                 "lmfish",
                 "Misk",
                 "octopus",
                 "pencil",
                 "personal_car",
                 "pocket",
                 "rat",
                 "ray",
                 "sea_snake",
                 "shoe",
                 "spoon",
                 "spring",
                 "stef",
                 "teddy",
                 "tree",
                 "truck",
                 "turtle",
                 "watch"};

map<string, int> Label;

void readtrain(string File1,string File2,vector<vector<Mat>> &list,vector<Mat> &label) {
    for (int i = 0; i < 70;++i)
        Label[ID[i]] = i;
    ifstream file1;
    file1.open(File1.data());
    assert(file1.is_open());
    for (int i = 0; i < 1125; ++i) {
        vector<Mat> tmp;
        Mat tmp_;
        tmp_.Init(resolution, resolution, 0);
        for (int x = 0; x < resolution; ++x) {
            for (int y = 0; y < resolution; ++y) {
                file1 >> tmp_.mat[x][y];
                tmp_.mat[x][y] /= 255.0;
            }
        }
        tmp.push_back(tmp_);
        list.push_back(tmp);
    }
    file1.close();

    ifstream file2;
    file2.open(File2.data());
    assert(file2.is_open());
    for (int i = 0; i < 1401; ++i) {
        Mat tmp_;
        tmp_.Init(1, 70, 0);
        string x;
        file2 >> x;
        tmp_.mat[0][Label[x]] = 1;
        label.push_back(tmp_);
    }
    file2.close();
}


void readval(string File1,string File2,vector<vector<Mat>> &list,vector<Mat> &label) {
    for (int i = 0; i < 70;++i)
        Label[ID[i]] = i;
    ifstream file1;
    file1.open(File1.data());
    assert(file1.is_open());
    for (int i = 0; i < 276; ++i) {
        vector<Mat> tmp;
        Mat tmp_;
        tmp_.Init(resolution, resolution, 0);
        for (int x = 0; x < resolution; ++x) {
            for (int y = 0; y < resolution; ++y) {
                file1 >> tmp_.mat[x][y];
                tmp_.mat[x][y] /= 255.0;
            }
        }
        tmp.push_back(tmp_);

        list.push_back(tmp);
    }
    file1.close();

    ifstream file2;
    file2.open(File2.data());
    assert(file2.is_open());
    for (int i = 0; i < 1401; ++i) {
        Mat tmp_;
        tmp_.Init(1, 70, 0);
        string x;
        file2 >> x;
        tmp_.mat[0][Label[x]] = 1;
        label.push_back(tmp_);
    }
    file2.close();
}


void SaveData (CNN &cnn) {

    puts("Saving...");

    ofstream fout;

    fout.open(Dataurl);

    //第一层卷积层的输入图像大小 卷积核大小 输入通道 输出通道
    fout << cnn.C1.InputH << " " << cnn.C1.InputW << " " << cnn.C1.Size << " " << cnn.C1.InChannels << " " << cnn.C1.OutChannels << endl;
    //第一层卷积层的卷积核
    for (int i = 0; i < cnn.C1.InChannels;++i) {
        for (int j = 0; j < cnn.C1.OutChannels;++j) {
            for (int r = 0; r < cnn.C1.Size;++r) {
                for (int c = 0; c < cnn.C1.Size;++c) {
                    fout << cnn.C1.Data[i][j].mat[r][c] << " ";
                }
                fout << endl;
            }
            fout << endl;
        }
        fout << endl;
    }
    fout << endl;
    //第一层卷积层的偏置
    for (int i = 0; i < cnn.C1.OutChannels;++i)
        fout<<cnn.C1.basicData.mat[0][i]<<" ";
    fout << endl;

    //第二层池化层的输入图像大小 卷积核大小 输入通道 输出通道 池化方式
    fout << cnn.S2.InputH << " " << cnn.S2.InputW << " " << cnn.S2.Size << " " << cnn.S2.InChannels << " " << cnn.S2.OutChannels << " " << cnn.S2.poolType << endl;

    //第三层卷积层的输入图像大小 卷积核大小 输入通道 输出通道
    fout << cnn.C3.InputH << " " << cnn.C3.InputW << " " << cnn.C3.Size << " " << cnn.C3.InChannels << " " << cnn.C3.OutChannels << endl;
    //第三层卷积层的卷积核
    for (int i = 0; i < cnn.C3.InChannels;++i) {
        for (int j = 0; j < cnn.C3.OutChannels;++j) {
            for (int r = 0; r < cnn.C3.Size;++r) {
                for (int c = 0; c < cnn.C3.Size;++c) {
                    fout << cnn.C3.Data[i][j].mat[r][c] << " ";
                }
                fout << endl;
            }
            fout << endl;
        }
        fout << endl;
    }
    fout << endl;
    //第三层卷积层的偏置
    for (int i = 0; i < cnn.C3.OutChannels;++i)
        fout<<cnn.C3.basicData.mat[0][i]<<" ";
    fout << endl;


    //第四层池化层的输入图像大小 卷积核大小 输入通道 输出通道 池化方式
    fout << cnn.S4.InputH << " " << cnn.S4.InputW << " " << cnn.S4.Size << " " << cnn.S4.InChannels << " " << cnn.S4.OutChannels << " " << cnn.S4.poolType << endl;

    //第三层卷积层的输入图像大小 卷积核大小 输入通道 输出通道
    fout << cnn.C5.InputH << " " << cnn.C5.InputW << " " << cnn.C5.Size << " " << cnn.C5.InChannels << " " << cnn.C5.OutChannels << endl;
    //第三层卷积层的卷积核
    for (int i = 0; i < cnn.C5.InChannels;++i) {
        for (int j = 0; j < cnn.C5.OutChannels;++j) {
            for (int r = 0; r < cnn.C5.Size;++r) {
                for (int c = 0; c < cnn.C5.Size;++c) {
                    fout << cnn.C5.Data[i][j].mat[r][c] << " ";
                }
                fout << endl;
            }
            fout << endl;
        }
        fout << endl;
    }
    fout << endl;
    //第三层卷积层的偏置
    for (int i = 0; i < cnn.C5.OutChannels;++i)
        fout<<cnn.C5.basicData.mat[0][i]<<" ";
    fout << endl;


    //第四层池化层的输入图像大小 卷积核大小 输入通道 输出通道 池化方式
    fout << cnn.S6.InputH << " " << cnn.S6.InputW << " " << cnn.S6.Size << " " << cnn.S6.InChannels << " " << cnn.S6.OutChannels << " " << cnn.S6.poolType << endl;


    //第七层输出层的输入通道大小 输出通道大小 偏置 权值
    fout << cnn.O7.inputNum << " " << cnn.O7.outputNum << endl;

    for (int i = 0; i < cnn.O7.outputNum;++i)
        fout<<cnn.O7.basicData.mat[0][i]<<" ";
    fout << endl;

    for (int i = 0; i < cnn.O7.outputNum;++i) {
        for (int j = 0; j < cnn.O7.inputNum;++j) {
            fout << cnn.O7.Data.mat[i][j] << " ";
        }
        fout << endl;
    }
    fout << endl;

    fout.close();

    puts("Saved");
}

vector<double>labels;
vector<vector<double>> images;


int FindMaxPos (Mat V) {
    double Maxn = -1.0;
    int Pos = 0;
    for (int i = 0; i < V.col;++i) {
        if (Maxn<V.mat[0][i]) {
            Maxn = V.mat[0][i];
            Pos = i;
        }
    }
    return Pos;
}

double Trues;

void Train (CNN &cnn, vector<vector<Mat>> inputData, vector<Mat> outputData, int trainNum, int trainTimes) {
    srand(time(0));
    cout << trainNum << endl;
    cnn.L.Init(1, trainNum, 0);
    int cnt = 0;
    stack<int> Stk;
    cnt = 0;
    clock_t StartTime = clock();
    for (int i = 0; i < trainNum;++i) { //训练图片数
        Learn = 0.005;
        work(cnn, inputData[i]);
        Back(cnn, outputData[i]);
        Update(cnn, inputData[i]);

        if (FindMaxPos (cnn.O7.outit) == FindMaxPos (outputData[i])) {
            ++cnt;
        }

        double l = 0;

        for (int j = 0; j < cnn.O7.outputNum;++j) {
            l -= outputData[i].mat[0][j] * log(cnn.O7.outit.mat[0][j]);
        }
        cnn.L.mat[0][i] = l;

        // if (l>=2.0)
        //     Stk.push(i);

        Clear(cnn);

        // if ((i+1)%1000==0)
        //     SaveData(cnn);
        cout << "epoch:[" << trainTimes << "] ";
        cout << "Photo: " << cnt << "/" << i + 1 << " ";
        clock_t EndTime = clock() - StartTime;
        double NowTime = (double)EndTime / CLOCKS_PER_SEC;
        double NowSpeed = (i + 1) / NowTime;
        printf("%.3lf", (double)cnt / (double)(i + 1) * 100.0);
        cout << "%";
        printf(" | L: %.3lf | t: %.3lf(s) | v: %.3lf(photo/s) | %.3lf(s) | %.3lf", cnn.L.mat[0][i], NowTime, NowSpeed, (trainNum - i) / NowSpeed,Trues);
        cout << "%" << endl;
    }

    // while (!Stk.empty ()) {
    //     int i = Stk.top();
    //     Stk.pop();
    //     Learn = 0.0005;
    //     work(cnn, inputData[i]);
    //     Back(cnn, outputData[i]);
    //     Update(cnn, inputData[i]);

    //     double l = 0;

    //     for (int j = 0; j < cnn.O7.outputNum;++j) {
    //         l -= outputData[i].mat[0][j] * log(cnn.O7.outit.mat[0][j]);
    //     }
    //     cnn.L.mat[0][i] = l;

    //     Clear(cnn);

    //     cout <<"L : "<<l<<" | "<< "Size : " << Stk.size() << endl;
    // }
    SaveData(cnn);
    // }
    puts("Over");
    return;
}

int Num[70][70];

void Test (CNN cnn, vector<vector<Mat>> inputData, vector<Mat> outputData) {
    int cnt = 0;
    clock_t StartTime = clock();
    for (int i = 0; i <inputData.size();++i) {
        work(cnn, inputData[i]);
        if (FindMaxPos (cnn.O7.outit) == FindMaxPos (outputData[i])) {
            ++cnt;
        }
        Clear(cnn);
        clock_t EndTime = clock() - StartTime;
        double NowTime = (double)EndTime / CLOCKS_PER_SEC;
        double NowSpeed = (i + 1) / NowTime;
        system ("cls");
        printf("Photo: %d | Used: %.3lf(s) | v: %.3lf(photo/s) | Need: %.3lf(s)\n", i + 1, NowTime, NowSpeed, (inputData.size() - i) / NowSpeed);
        cout << cnt << " " << i+1 << endl;
        cout<<100.0*(double)(cnt)/(double)(i+1)<<"%"<<endl;
        // cout<<"           ";
        // for (int x=0;x<70;++x) {
        //     cout<<ID[x]<<" ";
        // }
        // puts("");
        // for (int x = 0; x < 70;++x) {
        //     cout<<ID[x]<<" ";
        //     for (int y = 0; y < 70;++y) {
        //         printf("%-10d ", Num[x][y]);
        //     }
        //     puts("");
        // }
    }
    cout<<100.0*(double)(cnt)/(double)(inputData.size())<<"%"<<endl;
    Trues = 100.0 * (double)(cnt) / (double)(inputData.size());
    return;
}

void All_of_Work () {
    cout<<"start...."<<endl;
    vector<vector<Mat>> traindata_list;
    vector<Mat> traindata_label;
    vector<vector<Mat>> testdata_list;
    vector<Mat> testdata_label;
    srand (time (0));


    cout<<"start_reading_val....."<<endl;

    readval("C:\\Users\\64783\\CLionProjects\\untitled\\dataset_128x\\val\\dataset","C:\\Users\\64783\\CLionProjects\\untitled\\dataset_128x\\val\\label",testdata_list, testdata_label);
    cout<<"reading_val_over..."<<endl;


    int train_num = traindata_list.size ();
    int test_num = testdata_list.size();
    int inSize = 1;
    int outSize = testdata_label[0].col;
    int train_times = 30;

    // cout << inSize << " " << outSize << endl;

    // 每张图片为三通道 32*32
    int row = resolution;
    int col = resolution;

    CNN cnn;

    Init(cnn, row, col, inSize, outSize);
    puts("Init Over");
    while (train_times--) {
        traindata_list.resize(0);
        traindata_label.resize(0);
        cout<<"start_reading_train....."<<endl;
        readtrain("C:\\Users\\64783\\CLionProjects\\untitled\\dataset_128x\\train\\dataset","C:\\Users\\64783\\CLionProjects\\untitled\\dataset_128x\\train\\label",traindata_list, traindata_label);
        cout<<"reading_train_over..."<<endl;
        train_num = traindata_list.size ();
        Train(cnn, traindata_list, traindata_label, train_num, train_times);

        Test(cnn, testdata_list, testdata_label);

    }
    //Over
    return;
}
string fun (Mat Input_Photo) {
    CNN cnn;
    int row = Input_Photo.row;
    int col = Input_Photo.col;
    int inSize = 1;
    int outSize = 10;
    Init(cnn, row, col, inSize, outSize);
    vector<Mat> Input;
    Input.push_back(Input_Photo);
    work(cnn, Input);
    string Output = ID[FindMaxPos(cnn.O7.outit)];
    Clear(cnn);
    return Output;
}

int main () {
    All_of_Work();
    return 0;
}