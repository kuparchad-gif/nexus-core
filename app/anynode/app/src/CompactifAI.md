CompactifAI: Extreme Compression of Large Language Models  
 using Quantum-Inspired Tensor Networks  
 Andrei Tomut,1,2 Saeed S. Jahromi,3,1 Abhijoy Sarkar,1 Uygar Kurt,1 Sukhbinder  
 Singh,4 Faysal Ishtiaq,1 C´esar Mu˜noz,1 Prabdeep Singh Bajaj,1 Ali Elborady,1 Gianni  
 del Bimbo,1 Mehrazin Alizadeh,4 David Montero,1 Pablo Mart´ın-Ramiro,1 Muhammad  
 Ibrahim,1 Oussama Tahiri Alaoui,1 John Malcolm,4 Samuel Mugel,4 and Rom´an Or´us1,3,5,∗  
 arXiv:2401.14109v2  \[cs.CL\]  13 May 2024  
 1Multiverse Computing, Parque Cientifico y Tecnol´ogico de Gipuzkua,  
 Paseo de Miram´on, 170 3◦ Planta, 20014 Donostia / San Sebasti´an, Spain  
 2Catalan Institute of Nanoscience and Nanotechnology (ICN2),  
 CSIC and The Barcelona Institute of Science and Technology,  
 Campus UAB, 08193 Bellaterra, Catalonia, Spain.  
 3Donostia International Physics Center, Paseo Manuel de Lardizabal 4, E-20018 San Sebasti´an, Spain  
 4Multiverse Computing, Centre for Social Innovation,  
 192 Spadina Avenue Suite 509, Toronto, ON M5T 2C2 Canada  
 5Ikerbasque Foundation for Science, Maria Diaz de Haro 3, E-48013 Bilbao, Spain  
 Large Language Models (LLMs) such as ChatGPT and LlaMA are advancing rapidly in generative  
 Artificial Intelligence (AI), but their immense size poses significant challenges, such as huge training  
 and inference costs, substantial energy demands, and limitations for on-site deployment. Traditional  
 compression methods such as pruning, distillation, and low-rank approximation focus on reducing  
 the effective number of neurons in the network, while quantization focuses on reducing the numerical  
 precision of individual weights to reduce the model size while keeping the number of neurons fixed.  
 While these compression methods have been relatively successful in practice, there is no compelling  
 reason to believe that truncating the number of neurons is an optimal strategy. In this context, this  
 paper introduces CompactifAI, an innovative LLM compression approach using quantum-inspired  
 Tensor Networks that focuses on the model’s correlation space instead, allowing for a more controlled,  
 refined and interpretable model compression. Our method is versatile and can be implemented  
 with — or on top of — other compression techniques. As a benchmark, we demonstrate that a  
 combination of CompactifAI with quantization allows to reduce a 93% the memory size of LlaMA-2  
 7B, reducing also 70% the number of parameters, accelerating 50% the training and 25% the inference  
 times of the model, and just with a small accuracy drop of 2%- 3%, going much beyond of what  
 is achievable today by other compression techniques. Our methods also allow to perform a refined  
 layer sensitivity profiling, showing that deeper layers tend to be more suitable for tensor network  
 compression, which is compatible with recent observations on the ineffectiveness of those layers for  
 LLM performance. Our results imply that standard LLMs are, in fact, heavily overparametrized,  
 and do not need to be large at all.  
 Introduction.- The emergence of generative artificial in  
telligence (AI) has ushered in an era where computers can  
 perform tasks that were unimaginable just a few years  
 ago. A prime example of this advancement is found in  
 Large Language Models (LLMs) \[1\], which are based on  
 the innovative “transformer architecture.” \[2\] The field of  
 LLMs experienced a significant surge with the introduc  
tion of OpenAI’s ChatGPT \[3\], showcasing an unprece  
dented level of human-machine interaction. Following  
 this, several other models, such as Meta’s LlaMA \[4\] and  
 Google’s BERT \[5\], were developed. Currently, LLMs  
 are expected to be utilized not only in linguistic applica  
tions but also across various sectors, attracting substan  
tial investments in this transformative technology. This  
 development represents the most profound technological  
 revolution since the inception of the internet.  
 However, LLMs are not without their challenges. The  
 most significant issue is the energy consumption required  
 for training these AI models. As noted by the CEO of  
 OpenAI, training ChatGPT-3 incurred an estimated 100  
 million dollars in electricity bills alone, and the costs  
 for training such models are predicted to double every  
 ten months \[6\]. Coupled with the exponentially growing  
 demand for these systems, we face a daunting scenario:  
 the development of these systems is currently unsustain  
able without significantly impacting the planet. The im  
mense energy consumption of LLMs is untenable, com  
pelling the need for greener, more efficient solutions. In  
 this context, various compression techniques for LLMs  
 have been suggested, with quantization \[7\], distillation  
 \[8\] , pruning \[9\], and low-rank approximations \[10\] be  
ing among the most prominent. However, these methods  
 are quite brute-force — they largely focus on truncating  
 the effective number of neurons, even when the original  
 model’s accuracy is known to increase with size during  
 training. Consequently, controlling and anticipating the  
 compression error in these schemes is challenging, and  
 their application has met with mixed success.  
 In this paper, we introduce CompactifAI \[11\], a novel  
 LLM compression technique based on quantum-inspired  
2  
 Model  
 Size  
 Original 27.1 GB  
 Parameters Quantization  
 7B  
 8-bit  
 4-bit  
 88%  
 93%  
 6.8 GB  
 3.4 GB  
 4.1 GB  
 2.1 GB  
 7B  
 7B  
 2.1B  
 f  
 loat-32  
 int-8  
 int-4  
 f  
 loat-16  
 2.1B  
 mixed  
 TABLE I. Details of the models used in the benchmarks. The  
 quantization in the 93% compressed model is a mix of float-16  
 for the tensorized layers and int-4 quantization for the not  
tensorized layers.  
 Tensor Networks (TNs) \[12, 13\]. This technique involves  
 “tensorizing” the self-attention (SA) and multi-layer per  
ceptron (MLP) layers using a specific TN, which effec  
tively truncates the correlations present in the model.  
 The degree of truncation can be controlled via the bond  
 dimension of the TN, enabling a significant reduction  
 in the memory size and number of parameters of the  
 LLM model while maintaining accuracy. In practice,  
 the compressed model requires less energy and memory,  
 and operations such as training, retraining, and infer  
ence become more efficient. The “tensorized” model is  
 retrained using multi-GPU distributed training. Within  
 this framework, we observed that the significant reduc  
tion in the number of model parameters by tensorization  
 drastically reduces the GPU-CPU transfer time, conse  
quently reducing the training and inference time in our  
 benchmarks by 50% and 25%, respectively. Hence, our  
 tensorization approach is particularly well-suited for dis  
tributed training of LLMs. As we will demonstrate, a  
 brief retraining period allows the accuracy of the com  
pressed model to approach that of the original uncom  
pressed version.  
 Method.- The compression method we propose is based  
 on the efficient decomposition of weight matrices in neu  
ral networks into Tensor Networks, such as Matrix Prod  
uct Operators (MPOs) and similar structures. This con  
cept has been successfully implemented in deep learning  
 architectures, as previously demonstrated \[14–17\], but to  
 the best of our knowledge, this work is the first appli  
cation of this approach to compressing LLMs. Specifi  
cally for Large Language Models (LLMs), our approach  
 involves first, a layer sensitivity profiling (see Supplemen  
tary Information), that guides identifying the layers that  
 are more tenable to correlation compression and then  
 replacing their trainable weights with suitable TNs (in  
 the present case, MPOs). The results of our layer sensi  
tivity profiling are compatible with and in fact improve  
 (and refine) upon recent observations that deeper layers  
 tend to be ineffective in the performance of LLM mod  
els \[18\]. Without loss of generality, here we consider the  
 case of the LLM architecture of LlaMA-2 chat models.  
 As illustrated in Fig.1, we substitute the weight matrices  
 (a)   
(b)   
Embedding   
Layer  
 Embeding   
Layer  
 Tokenized Input  
 Tokenized Input  
 Reshape  
 SVD  
 LlaMA Decoder Block  
 Llama Decoder Block   
Head   
Layer  
 Head  
 Layer  
 . . . .  
 SA     MLP  
 SA MLP  
 SA MLP  
 SA MLP  
 Tokenized Output  
 Tokenized Output  
 FIG. 1\. \[Color online\] (a) Example, in diagrammatic repre  
sentation, of the decomposition of a weight matrix W in terms  
 of an MPO. The original matrix has 216 × 216 parameters.  
 After reshaping the matrix indices followed by two sequential  
 SVDs, the resulting tensor network has 2 × 36χ \+ 36χ2 pa  
rameters, amounting to the sum of parameters of each tensor,  
 with χ being the MPObonddimensionserving as a truncation  
 parameter. In the diagrammatic representation of MPOs, cir  
cles represent individual tensors, lines indicate tensor indices  
 and lines connecting circles represent contracted shared in  
dices between tensors. (b) Schematic graphical representation  
 of the tensorization mechanism for LLMs within the LlaMA  
 model family (generalization to other LLM architectures is  
 straightforward). Embedding and head layers customize the  
 input and output for a given task, with the tokenized in  
put/output comprising words and sentences. The Self At  
tention and Multi-layer Perceptron layers within the LlaMA  
 decoder block are tensorized in such a way that the weight ma  
trices of the corresponding neural networks are decomposed  
 into appropriate Tensor Networks, in this case, MPOs with a  
 bond dimension of χ.  
 in the Self Attention (SA) and Multi-layer Perceptron  
 (MLP) layers of the (pretrained) LlaMA decoder block  
 with MPOs characterized by a bond dimension χ. The  
 process of determining the MPO involves executing se  
quential Singular Value Decompositions (SVDs) on the  
 respective weight matrix, retaining the largest χ singular  
 values at each SVD. This truncation in χ effectively trun  
cates the correlations among model parameters within a  
 given layer to only the most relevant ones necessary to  
 describe the system, while discarding those that are ir  
relevant. The approach leads to a significant reduction  
 in memory costs, as storing the truncated MPO, which  
 incurs a polynomial cost, is far more efficient than storing  
 the original weight matrix, which would require an expo  
nential cost in the number of neurons within the layer.  
 Furthermore, the bond dimension χ effectively controls  
 the level of compression: a smaller χ results in more in  
formation being discarded, leading to greater compres  
sion but at the cost of reduced accuracy. The choice of  
 TN architecture as well as the number of decomposed  
 tensors for each layer can be considered as additional  
3  
 Accuracy	(%)	  
10  
 20  
 30  
 40  
 50  
 60  
 70  
 80  
 90  
	  
MMLU  
 HellaSwag  
 BoolQ  
 TriviaQA  
 GSM8K  
	  
Original  
 8	Bit  
 4	Bit  
 88%	Compressed  
 93%	Compressed  
 FIG. 2\. \[Color online\]Accuracies of theoriginal andcom  
pressedmodelsforthetasksrelatedtoLanguageUnderstand  
ing(MMLU),Commonsensereasoning(HellaSwag),Reading  
 comprehension (BoolQ),Worldknowledge (TriviaQA) and  
 Math(GSM8K).Theaccuracyofthecompressedmodelsonly  
 deviatesby2%-3%comparedtotheoriginalLlaMA27B.  
 Training	Time	(m)  
 10  
 12  
 14  
 16  
 18  
 20  
 22  
	  
Original  
 8	Bit  
 4	Bit  
 88%	Compressed  
 93%	Compressed  
	  
FIG.3. \[Coloronline\]Trainingtime(inminutes)of thedif  
ferentmodelsonthesameamountofMMLUdatausedfor  
 healingthetensorizedmodels. Thetensorizedmodels show  
 2xspeedup(i.e. half thetime)withdistributedtrainingon  
 eightA10gNVIDIAGPUswithrespecttoboththeoriginal  
 andpurely-quantizedmodels.  
 hyper-parametersforthecompressedmodel.  
 Toensurehighaccuracyinthecompressedmodel,our  
 methodalso includesarapidretrainingphase, dubbed  
 ashealing, followingthedeterminationof thetruncated  
 MPOs. This retraining is essential because the local,  
 layer-by-layer truncation intoMPOs–akin to the so  
called“simpleupdate”inTNalgorithms \[19\]–maynot  
 Inference	Time	(ms)  
 0  
 0.2  
 0.4  
 0.6  
 0.8  
 1  
 1.2  
	  
MMLU  
 HellaSwag  
 BoolQ  
 TriviaQA  
 GSM8K  
	  
Original							8	Bit							4	Bit  
 88%	Compressed							93%	Compressed  
 FIG.4. \[Coloronline\] Inferencetime(inmilliseconds)ofthe  
 differentmodels formeasuringtheaccuraciesof theMMLU,  
 HellaSwag,BoolQ,TriviaQA, andGSM8Ktasks. The ten  
sorizedmodelsare25%fasterwithdistributedinferenceon  
 eightA10gNVIDIAGPUswithrespecttotheoriginalmodel.  
 Noticethat inferencetimeof somequantizedmodels iseven  
 higherthanthatoftheoriginal.Theinferencetimesarenor  
malizedbytakingthetimeoftheoriginalmodelasareference.  
 beoptimalingeneral, inthesensethattheeffectofother  
 layersarenot explicitlytaken intoaccountwhentrun  
cating theweightmatrixof a specific layer. However,  
 retrainingthecompressedstructureiswaymoreefficient  
 thantrainingtheoriginaluncompressedmodelduetothe  
 significantlysmallernumberofmodelparameters,which  
 reduces theCPU-GPUtransfer times inadistributed  
 trainingsetup. Aswedemonstratebelow, after justa  
 fewretrainingepochsofthecompressedmodel, itsaccu  
racycloselyapproachesthatoftheoriginaluncompressed  
 modelbutatafractionofthecost.  
 Benchmark.-Toevaluateourmethod,weused it to  
 compresstheLlaMA-27Bmodel.Thismodelrepresents  
 the“smallest”withinthe“large”categoryofLLMs in  
 theopen-sourceLlaMAseries, developedbyMETA. It  
 encompasses7billionparameters, hasbeenpre-trained  
 onover2trilliontokens,offersacontext lengthof4096,  
 andhasundergonefine-tuningwithmorethan1million  
 humanannotations.  
 Inorder tobenchmarkthemodelwecreatedseveral  
 compressedversionsoftheLlaMA-27Bmodelsbyacom  
binationoftensornetworkcompressionandquantization.  
 Detailsofthebenchmarkedmodels,suchastheirsizein  
 memory, number of parameters andtheir quantization  
 areavailable inTable I.The8-bitand4-bitquantized  
 modelswerecreatedfromtheoriginalLlaMAmodelby  
 usingthebitsandbytesquantization library. Further  
more,the88%and93%compressedmodelswerecreated  
4  
 Task/Model Original 8-bit 4-bit 88% 93%  
 MMLU  
 46.41  
 HellaSwag  
 BoolQ  
 TriviaQ  
 GSM8K  
 80.55  
 79.76  
 19.03  
 23.05  
 46.03 45.53 45.32 44.16  
 79.77 79.25 77.87 76.54  
 78.81 78.19 77.90 76.77  
 19.01 19.00 18.33 18.10  
 22.71 22.44 22.58 17.74  
 TABLEII. Accuracies of the models in Table I for the MMLU,  
 HellaSwag, BoolQ, TriviaQA and GSM8K tasks.  
 by applying CompactifAI to the float-16 quantized ver  
sion of the original LlaMA model. The parameter counts  
 in Table I show that while the size of the models can be  
 reduced both by tensor network compression and quan  
tization, the tensorized model allows for more size reduc  
tion than quantization by significantly reducing the model  
 parameters. As we will show, unlike in quantization, this  
 parameter reduction is the key feature that allows signif  
icant speed up in both training and inference.  
 Let us further note that the tensorized models in Ta  
ble I were healed after compression to recover the ac  
curacy drop caused by parameter reduction. We used  
 generic chat datasets such as Ultrachat, Alpaca and Open  
Hermess to retrain the tensorized model, which was im  
plemented on a single AWS EC2 instance with 8 NVIDIA  
 A10g GPU processors and distributed training. The  
 healing process was performed for less than a single epoch  
 on the aforementioned datasets. The 93% compressed  
 model was obtained by applying 4-bit quantization to the  
 not-tensorized layers of the 88% compressed and healed  
 model.  
 Once the compressed models are healed, we bench  
marked all the models of Table I in tasks related to  
 language understanding (MMLU), commonsense reason  
ing (HellaSwag), reading comprehension (BoolQ), world  
 knowledge (TriviaQA) and math (GSM8K). We further  
 used the LLM Evaluation Harness \[20\] library to calcu  
late the accuracies on these five tasks.  
 Fig. 2 shows the accuracy of the original and com  
pressed models for the target tasks. While the accuracies  
 of all compressed models are very close to those of the  
 original 7B LlaMA model and only deviate by 2%−3%,  
 the 88% and 93%compressed models reach the same level  
 of accuracies with 70% fewer parameters (just 2.1 billion).  
 This suggests that a substantial portion of the parame  
ters in LLMs are redundant, and discarding even up to  
 70% of the parameters does not degrade the model ac  
curacy significantly. Such behavior suggests that, after  
 all, large language models are heavily overparametrized,  
 and they do not need to be large in practice. For bet  
ter clarity, we have further reported all the accuracies in  
 Table II for the original and compressed models. Let us  
 stress that the accuracies of the tensorized models are ob  
tained by training only for one epoch during the healing,  
 and better accuracies can be obtained with further fine  
tuning of the tensorized models, sometimes even higher  
 than that of the original, as we have seen in parallel tests  
 with smaller models.  
 On top of the accuracy results, another interesting ob  
servation that demonstrates the power of tensor network  
 compression is the remarkable speedup that tensorized  
 models manifest both during training and inference. In  
 order to benchmark the training speed of all models in  
 Table I, we trained the original and the quantized models  
 on the same amount of data used for tensorized mod  
els, and measured their training times. Fig. 3 shows  
 the training times of all models compared against each  
 other. Tensorized models exhibit a remarkable 50% ac  
celeration (i.e. 2x faster) compared to the original and  
 purely quantized models. This significant speedup is at  
tributed to the substantially smaller number of param  
eters in tensorized models, which are transferred much  
 faster between the GPUs and CPUs during distributed  
 training.  
 Next, we tested all models for inference time using dis  
tributed training with both data and model paralleliza  
tion. Fig. 4 shows the inference time (forward time of  
 the model) for different models compared to the original  
 model. Inference times are in milliseconds, and normal  
ized with respect to that of the original model (dark blue  
 bar in the plot). Let us point out that while tensorized  
 models bring more than 25% speed up in the inference,  
 the 4-bit quantized model, on the contrary, slows down  
 the inference by 13%. This might be due to the fact that  
 some quantized operations cannot be processed efficiently  
 on conventional generations of GPUs. LlaMA-2 7B  
 Conclusions.- In this paper, we have introduced and  
 benchmarked CompactifAI, a compression method of  
 Large Language Models based on quantum-inspired Ten  
sor Networks. The method decomposes weight matrices  
 in Self Attention and Multi-layer Perceptron layers of  
 the LLM in terms of Matrix Product Operators with a  
 given bond dimension, effectively truncating in the cor  
relations present in the system. The compression rate  
 of the model can be controlled via the bond dimension  
 and the model accuracy can be ramped up with a short  
 retraining (healing) process. We have shown that a com  
bination of CompactifAI with quantization allows to re  
duce a 93% the memory size of LlaMA-2 7B, reducing  
 also 70% the number of parameters, accelerating 50% the  
 training and 25% the inference times of the model, and  
 just with a small accuracy drop of 2%- 3%. This goes  
 much beyond of what is achievable by other compression  
 techniques and also shows that standard LLMs are, in  
 fact, heavily overparametrized, something that “natural  
 intelligence” is definitely not doing.  
 Our work provides a much more refined, controllable,  
 and explainable compression technique of LLMs com  
pared to alternative methods such as pruning, distilla  
5  
 tion, quantization, and low-rank approximations. Fur  
thermore, our TN method is compatible with all these  
 techniques and can be applied alongside with them, as  
 we have shown in this paper. This can be further ex  
plored in future works, as well as more advanced TN  
 compression techniques for LLMs.  
 In our opinion, our work opens the door to the de  
mocratization of LLMs. Small-size LLMs are a necessity  
 to lower their gargantuan energy consumption, and can  
 also be deployed on premises without the need of a cloud  
 connection to somebody else’s server, in turn opening  
 an entire new world of possibilities for personalized-AI  
 models. In our opinion, CompactifAI and tensor net  
work methods are going to play a fundamental role in  
 the development of the next generation of AI technology.  
 Acknowledgements: We acknowledge Donostia In  
ternational Physics Center (DIPC), Ikerbasque, Basque  
 Government, Diputaci´on de Gipuzkoa, European Inno  
vation Council (EIC), and Spanish Government for con  
stant support, as well as insightful discussions with the  
 team from Multiverse Computing. A. T. acknowledges  
 funding by MICIN within NextGenerationEU(PRTR  
C17.I1) program and by Generalitat de Catalunya. S.  
 S. J. also acknowledges the Institute for Advanced Stud  
ies in Basic Sciences (IASBS).  
 Data availability statement: all data required for  
 this project can be accessed upon reasonable request by  
 contacting the authors.  
 ∗ roman.orus@multiversecomputing.com  
 \[1\] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and  
 I. Sutskever, Language Models are Unsupervised Multi  
task Learners, OpenAI Technical Report (2019).  
 \[2\] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit,  
 L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin,  
 Attention is all you need, in Advances in Neural Infor  
mation Processing Systems, Vol. 30, edited by I. Guyon,  
 U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vish  
wanathan, and R. Garnett (Curran Associates, Inc.,  
 2017).  
 \[3\] S. Lock, What is AI chatbot phenomenon ChatGPT and  
 could it replace humans?, The Guardian (2022).  
 \[4\] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A.  
 Lachaux, T. Lacroix, B. Rozi\`ere, N. Goyal, E. Ham  
bro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave,  
 and G. Lample, LlaMA: Open and Efficient Foundation  
 Language Models 10.48550/arXiv.2302.13971 (2023),  
 arXiv:2302.13971.  
 \[5\] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova,  
 BERT: Pre-training of Deep Bidirectional Transformers  
 for Language Understanding 10.48550/arXiv.1810.04805  
 (2018), arXiv:1810.04805.  
 \[6\] The bigger-is-better approach to AI is running out of  
 road, The Economist (2023).  
 \[7\] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. G.  
 Howard, H. Adam, and D. Kalenichenko, Quantization  
 and training of neural networks for efficient integer  
arithmetic-only inference, 2018 IEEE/CVF Conference  
 on Computer Vision and Pattern Recognition , 2704  
 (2017).  
 \[8\] G. E. Hinton, O. Vinyals, and J. Dean, Distilling the  
 knowledge in a neural network, ArXiv abs/1503.02531  
 (2015).  
 \[9\] S. Han, J. Pool, J. Tran, and W. J. Dally, Learning both  
 weights and connections for efficient neural network, in  
 Neural Information Processing Systems (2015).  
 \[10\] M. Jaderberg, A. Vedaldi, and A. Zisserman, Speeding up  
 convolutional neural networks with low rank expansions,  
 ArXiv abs/1405.3866 (2014).  
 \[11\] See also, Multiverse Computing CompactifAI (2023).  
 \[12\] R. Or´us, A practical introduction to tensor net  
works: Matrix product states and projected entan  
gled pair states, Annals of Physics 349, 117 (2014),  
 arXiv:1306.2164.  
 \[13\] R. Or´us, Tensor networks for complex quantum systems,  
 Nature Reviews Physics 1, 538 (2019).  
 \[14\] A. Novikov, D. Podoprikhin, A. Osokin, and D. Vetrov,  
 Tensorizing neural networks (2015), arXiv:1509.06569  
 \[cs.LG\].  
 \[15\] R. Patel, C.-W. Hsing, S. Sahin, S. S. Jahromi, S. Palmer,  
 S. Sharma, C. Michel, V. Porte, M. Abid, S. Aubert,  
 P. Castellani, C.-G. Lee, S. Mugel, and R. Or´us,  
 Quantum-Inspired Tensor Neural Networks for Par  
tial Differential Equations 10.48550/arXiv.2208.02235  
 (2022), arXiv:2208.02235.  
 \[16\] S. S. Jahromi and R. Or´us, Variational tensor neural net  
works for deep learning, arXiv preprint arXiv:2211.14657  
 (2022).  
 \[17\] M. Wang, Y. Pan, Z. Xu, X. Yang, G. Li, and A. Ci  
chocki, Tensor networks meet neural networks: A survey  
 and future perspectives (2023), arXiv:2302.09019 \[cs.LG\].  
 \[18\] A. Gromov, K. Tirumala, H. Shapourian, P. Glorioso,  
 and D. A. Roberts, The unreasonable ineffectiveness of  
 the deeper layers (2024), arXiv:2403.17887 \[cs.CL\].  
 \[19\] H. C. Jiang, Z. Y. Weng, and T. Xiang, Accurate de  
termination of tensor network state of quantum lattice  
 models in two dimensions, Phys. Rev. Lett. 101, 090603  
 (2008).  
 \[20\] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black,  
 A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h,  
 H. Li, K. McDonell, N. Muennighoff, C. Ociepa,  
 J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron,  
 L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang,  
 and A. Zou, A framework for few-shot language model  
 evaluation (2023).  
6  
 SUPPLEMENTARYINFORMATION:LAYERSENSITIVITYPROFILINGOFLLAMA-27B  
 MMLU	Total	Accuracy  
 0.2  
 0.25  
 0.3  
 0.35  
 0.4  
 0.45  
 0.5  
	  
Dmax  
 10 20 30 40 50 60 70 80 90  
 Block\[15\]  
 L1  
 L2  
 L3  
 L4  
 L5  
 L6  
 L7  
 Original  
 MMLU	Total	Accuracy  
 0.2  
 0.25  
 0.3  
 0.35  
 0.4  
 0.45  
 0.5  
	  
Dmax  
 10 20 30 40 50 60 70 80 90  
 Block\[31\]  
 L1  
 L2  
 L3  
 L4  
 L5  
 L6  
 L7  
 F1  
 L1  
 L2  
 L3  
 L4  
 L5  
 L6  
 L7  
 Original  
 MMLU	Total	Accuracy  
 0.2  
 0.25  
 0.3  
 0.35  
 0.4  
 0.45  
 0.5  
	  
Dmax  
 10 20 30 40 50 60 70 80 90  
 Block\[0\]  
 L1  
 L2  
 L3  
 L4  
 L5  
 L6  
 L7  
 Original  
 MMLU	Total	Accuracy  
 0.2  
 0.25  
 0.3  
 0.35  
 0.4  
 0.45  
 0.5  
	  
Dmax  
 10 20 30 40 50 60 70 80 90  
 Block\[5\]  
 L1  
 L2  
 L3  
 L4  
 L5  
 L6  
 L7  
 F1  
 L1  
 L2  
 L3  
 L4  
 L5  
 L6  
 L7  
 Original  
 FIG.5. \[Coloronline\]Layersensitivityanalysisfor layers inseveralattentionblocksoftheLlaMA-27Bmodel. LayersL1-L4  
 correspondtothemulti-headattentionlayersandL5-L7areMLPlayers.L7isthelastlayerwhichalsoaccountsfortheoutput  
 of theattentionblock. TheLayersof the initialblocksaremoresensitivetocompression. However, themiddletotheend  
 blocksarerobusttoverylargecompressions.  
 ThemainbuildingblocksofLlaMA-27Bmodelare32attentionblockseachofwhichiscomposedof fourmulti  
headattentionlayersandthreemulti-layerperceptronlayers.Dependingonthelocationoftheattentionblocks, its  
 internal layershavedifferentsensitivitytotensornetworkdecompositions.Tohavethemostefficienttensornetwork  
 compression,wedevelopedatool forprofilingthelayersoftheLLMtoassessitssensitivitytothecompressionlevel.  
 Fig. 5showstheprofilingresults forthetotalMMLUaccuracyversusthemaximumtensorbonddimensionsDmax  
 thatemergeduringthetensordecompositionoflayerweights.Wehaveshownresultsforfourdifferentblocksfromthe  
 beginning, themiddle,andattheendoftheLlaMAnetworkofattentionblocks, i.e.,blocks0,5,15,and31. Letus  
 furthernotethatasthebonddimensionbecomessmaller,morecompressionhappenswhichimpliesmoreparameters  
 werediscardedduringtensordecompositionofweights.  
 Fig. 5shows that for theLlaMA-27Bmodel, the initial layersandattentionblockswhichare locatedat the  
 beginningoftheattentionnetworkaremoresensitivetotruncationandcompression.However,aswemovetowards  
 theendof thenetwork, thesensitivitydecreasesandwecancompress the layersdownto10%of theoriginal size  
 withoutsignificant lossofaccuracy.This, inturn, iscompatiblewithpreviousobservationsontheineffectivenessof  
 deeperlayers\[18\].Theresultssuggeststhattheattentionblockstowardsthemiddletoendoftheattentionnetworks  
7  
 are more suitable for large-scale tensorization and truncation. However, the layers at the beginning should be handled  
 with more care and it is advised not to compress them below 50%. Another observation from Fig.5 is that the last  
 MLP layer in each attention block, which amounts to the output of the block, is more sensitive to compression in  
 the LlaMA-2 7B model. It is therefore advised that this layer be excluded from tensorization and compression in all  
 attention blocks.  
 Last but not least, we have used the MMLU accuracy as the measure of layer sensitivity of the layers. This can be  
 done by using any dataset and metric which is relevant to the underlying model. The tensorized models in this work  
 are all created after such a layer analysis to obtain the best-performing model