> 以 2.14 代码路径 -> 99d80a9e254c9df7940b2902b14d15914dbbbcd9
- [xla jit 相关的路径](#xla-jit-相关的路径)
- [jit 相关的 pass 详解](#jit-相关的-pass-详解)
  - [EncapsulateXlaComputationsPass::Run](#encapsulatexlacomputationspassrun)
  - [CloneConstantsForBetterClusteringPass](#cloneconstantsforbetterclusteringpass)
  - [ClusterScopingPass::Run](#clusterscopingpassrun)
  - [MarkForCompilationPass::Run](#markforcompilationpassrun)
    - [RegisterCompilationKernels](#registercompilationkernels)
    - [Initialize get all the clusters](#initialize-get-all-the-clusters)
    - [DeclusterNodes](#declusternodes)
    - [CreateClusters](#createclusters)
  - [ForeXlaConstantsOnHostPass::Run](#forexlaconstantsonhostpassrun)
  - [IncreaseDynamismForAutoJitPass::Run](#increasedynamismforautojitpassrun)
  - [PartiallyDeclusterPass::Run](#partiallydeclusterpassrun)
  - [ReportClusteringInfoPass::Run](#reportclusteringinfopassrun)
  - [EncapsulateSubgraphsPass::Run](#encapsulatesubgraphspassrun)
  - [BuildXlaOpsPass::Run](#buildxlaopspassrun)
- [XlaCompileOp](#xlacompileop)

# xla jit 相关的路径

[jit:](tensorflow/compiler/jit)

<!-- [tf2xla:](tensorflow/compiler/tf2xla) -->

<!-- [xla_client](tensorflow/compiler/xla/client) -->

<!-- [xla/service](tensorflow/compiler/xla/service) -->

# jit 相关的 pass 详解
jit 流程是注册了10多个pass

主要是掌握一下底下的多个pass
## [EncapsulateXlaComputationsPass::Run](tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc:380)
  - [Encapsulate](tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc:197)TODO:看一下 src dst等 是否有`kXlaClusterIdAttr` 和 `kXlaClusterOutput` 属性; 含有相同的 `_xla_compile_id`的节点聚合在一起
    - [EncapsulateSubgraphsInFunctions](tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc:221)
      - [SplitIntoSubgraphs](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1093) TODO: `group_attribute` 相同的节点进行聚合, 分裂成子图, 然后用 func calls 来进行替换
        - [CopySubgraphNodes](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:765) 拷贝所有设置了 `group_attribute`的节点
        - [CopySubgraphEdges](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:766)  拷贝所有设置了 `group_att`的边, 添加 `_Arg & _Retval` 属性到子图之间
        - [MarkGuaranteedConstants](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:767)  子图中的节点, 如何parents全是 const 就 mark 一下, 添加`_is_guaranteed_constant`属性, 在 `RewriteSubgraph` 中用到
      - [BuildFunctionDefs](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1095) TODO:重写图到funcdef
        - [RewriteSubgraph](tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc:114) 重写图,主要是保证稳定性? 输入和输出的permutation,按照`resource,name`的 pair 来排序
        - [BuildControlFlowInfo](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:603) 看 function 的依赖
        - [GraphToFunctionDef](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:604) 把图转换成funcdef 
        - [library->AddFunctionDef](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:615)  有一个 func 的缓存可以用
      - [BuildOutputGraph](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1100)  
  - [BuildXlaLaunchOps](tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc:397)
    

  - [BuildXlaLaunchOps](tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc:361)
  - 这个的作用主要是把 python 端赋值的 `_xla_compile_id` 转换过来, 多加点 log 看下 TODO:

## [CloneConstantsForBetterClusteringPass](tensorflow/compiler/jit/clone_constants_for_better_clustering.h:54)
  - [CloneConstantsForBetterClusteringPassImpl::Run](tensorflow/compiler/jit/clone_constants_for_better_clustering.cc:143)
  - 把小的constant clone一遍, 减少不同的 cluster 之间的等待?
  - 具体的实现是直接修改的图, dumplicate添加新的节点, 移除老的节点

## [ClusterScopingPass::Run](tensorflow/compiler/jit/cluster_scoping_pass.cc:157)
  - 添加`_XlaInternalScope`信息, 怎么用的? 最终为了方便并行?
  - 这个是用来添加 stage 的 pass, TensorFlow 中，`Stage` 和 `Unstage` 是两种特殊的操作，它们用于实现数据的pipeline并行处理
  - `Stage` 指的cpu 将操作放入到 gpu 可以直接读取的区域, 这之前的所有 op 加一个scope
  - `Unstage` 指的是 cpu 直接读取 gpu 的结果, 这之后的所有 op 加一个scope, 防止被 cluster 到一起



## [MarkForCompilationPass::Run](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1907)
  - 动作
    - 用于聚合出来`_XlaCluster=<cluster id>`
    - 负责寻找合适的`Cluster`, 并为找到的同一个`Cluster`内所有节点设置同样的`_xla_compile_id`属性
  - [GetMarkForCompilationPassFlags](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1909)  
    - parse flags, such as `max_cluster_size` ...
    - [Compiler fuel](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1921)
      - `fuel`是一个用于控制优化过程的参数。 具体来说，它限制了可以进行的优化操作的数量。每进行一次优化操作， 燃料就会减少一点。 当燃料耗尽（即值为0）时， 优化过程就会停止, 这里默认是无穷
  - [MarkForCompilation](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1924)
    - [FixupSourceAndSinkEdges](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1861)
      - connect all nodes to source / sink node, -> deadness analysis
### [RegisterCompilationKernels](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1588)
- register all kernels that can be compile cpu + gpu #TODO  

### [Initialize](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1594) get all the clusters
- [FindCompilationCandidates](tensorflow/compiler/jit/mark_for_compilation_pass.cc:658)
  - [BackwardsConstAnalysis](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1274) 分析 const 变量, 方便后面进行优化
  - for node in nodes
    - 获取 allowlist, registeredOp, get device 
    - TODO: compile time const 分析
    - [IsIdentityDrivingConstsInLoop](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1463) 
    - 分解 while / for loop 的边
    - if ok add to compilation_candidates_
- [CreateCycleDetectionGraph](tensorflow/compiler/jit/mark_for_compilation_pass.cc:666) TODO: 检查是否有环形的图, 为了后面的处理
- [DeadnessAnalysis::Run](tensorflow/compiler/jit/mark_for_compilation_pass.cc:675)  TODO: 移除掉已经 dead 的图
- [BuildInitialClusterSet](tensorflow/compiler/jit/mark_for_compilation_pass.cc:687) 
[RunEdgeContractionLoop](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1601)
- > edge contraction 参考 (https://rockt.github.io/2018/04/30/einsum)        
- TODO:
### [DeclusterNodes](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1602) 
- 用于去除掉fill 节点,用于在计算图中填充固定值, 在 cluster 里面实例化太晚了, 比如 `tf.fill` op?  
### [CreateClusters](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1603)
- [ShouldCompileCluster](tensorflow/compiler/jit/mark_for_compilation_pass.cc:978) 里面有个 cache
  - [PickDeviceForXla](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1792) //_ pick 一个 device, 里面有 cache, 这也是最容易出错的地方..
  - [ShouldCompile](tensorflow/compiler/jit/mark_for_compilation_pass.cc:1806)// 检查一些属性 






## [ForeXlaConstantsOnHostPass::Run](tensorflow/compiler/jit/force_xla_constants_on_host_pass.cc:24)
## [IncreaseDynamismForAutoJitPass::Run](tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc:375)
## [PartiallyDeclusterPass::Run](tensorflow/compiler/jit/partially_decluster_pass.cc:405)
  - 上面三个对 `mark` 结果进行微调
## [ReportClusteringInfoPass::Run](tensorflow/compiler/jit/report_clustering_info_pass.cc:23)
## [EncapsulateSubgraphsPass::Run](tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1140)
  - ""将每个cluster 内的多个节点换成单个节点,记录子图信息
## [BuildXlaOpsPass::Run](tensorflow/compiler/jit/build_xla_ops_pass.cc:571)
  - 替换融合后的节点到 `XlaCompileOp` + `XlaRunOp`两个算子


# XlaCompileOp
`XlaCompileOp` 用来进行 `Cluster`的所有输入以及子图信息，在运行时进行编译（编译存在缓存）

- [XlaCompileOp::Compute](tensorflow/compiler/jit/kernels/xla_ops.cc:783)
  - [CompileToLocalExecutable](tensorflow/compiler/jit/kernels/xla_ops.cc:381)
    - [CompileIfNeeded](tensorflow/compiler/jit/device_compiler.h:245)
      - [CompileImpl](tensorflow/compiler/jit/device_compiler.h:395)
      - [CompileAsynchronous](tensorflow/compiler/jit/device_compiler.h:350)
        - [CompileStrict](tensorflow/compiler/jit/device_compiler.h:280)
          - [TfGraphToHloCompiler::Compile](tensorflow/compiler/jit/tf_graph_to_hlo_compiler.cc:22)
            - [XlaCompiler::CompileFunction](tensorflow/compiler/tf2xla/xla_compiler.cc:801)
              - [XlaCompiler::CompileGraph](tensorflow/compiler/tf2xla/xla_compiler.cc:1430)
                - [ExecuteGraph](tensorflow/compiler/tf2xla/xla_compiler.cc:139)
              - [CompileGraphToXlaHlo](tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.cc:959)
          - [BuildExecutable](#BuildExecutable) 
            - [LocalClient::Compile](tensorflow/compiler/xla/client/local_client.cc:386)
              - 最终得到 `LocalExecutable`
