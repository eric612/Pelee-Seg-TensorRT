#include "pluginImplement.h"
#include "mathFunctions.h"
#include <vector>
#include <algorithm>





/******************************/
// PluginFactory //
/******************************/
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    assert(isPlugin(layerName));

    if (!strcmp(layerName, "ext/pm1_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm1_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm1_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm1_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm1_mbox_conf_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm1_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm1_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm1_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm2_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm2_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm2_mbox_conf_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm2_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm2_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm3_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm3_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm3_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm3_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm3_mbox_conf_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm3_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm3_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm3_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm4_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm4_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm4_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm4_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm4_mbox_conf_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm4_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm4_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm4_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm5_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm5_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm5_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm5_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm5_mbox_conf_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm5_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm5_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm5_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm6_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm6_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm6_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm6_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm6_mbox_conf_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm6_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm6_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mExt_pm6_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm1_mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm1_mbox_priorbox_layer.get() == nullptr);
        PriorBoxParameters params; 
        float min_size[1] = {30.3999996185}, max_size[1] = {60.7999992371}, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        params.minSize=min_size;
        params.aspectRatios=aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 3;
        params.maxSize = max_size;
        params.numMaxSize = 1;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mExt_pm1_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mExt_pm1_mbox_priorbox_layer.get();
    }
      else if (!strcmp(layerName, "ext/pm2_mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm2_mbox_priorbox_layer.get() == nullptr);
        float min_size[1] = {60.7999992371}, max_size[1] = {112.480003357}, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        PriorBoxParameters params; 
        params.minSize=min_size;
        params.aspectRatios=aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 3;
        params.maxSize = max_size;
        params.numMaxSize = 1;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;



        mExt_pm2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mExt_pm2_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm3_mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm3_mbox_priorbox_layer.get() == nullptr);
        float min_size[1] = {112.480003357}, max_size[1] = {164.160003662}, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        PriorBoxParameters params; 
        params.minSize=min_size;
        params.aspectRatios=aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 3;
        params.maxSize = max_size;
        params.numMaxSize = 1;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;

        mExt_pm3_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mExt_pm3_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm4_mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm4_mbox_priorbox_layer.get() == nullptr);
        float min_size[1] = {164.160003662}, max_size[1] = {215.839996338}, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        PriorBoxParameters params; 
        params.minSize=min_size;
        params.aspectRatios=aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 3;
        params.maxSize = max_size;
        params.numMaxSize = 1;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mExt_pm4_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mExt_pm4_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm5_mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm5_mbox_priorbox_layer.get() == nullptr);
        float min_size[1]= {215.839996338}, max_size[1]= {267.519989014}, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        PriorBoxParameters params; 
        params.minSize=min_size;
        params.aspectRatios=aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 3;
        params.maxSize = max_size;
        params.numMaxSize = 1;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mExt_pm5_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mExt_pm5_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm6_mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm6_mbox_priorbox_layer.get() == nullptr);
        float min_size[1] = {267.519989014}, max_size[1] = {319.200012207}, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        PriorBoxParameters params; 
        params.minSize=min_size;
        params.aspectRatios=aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 3;
        params.maxSize = max_size;
        params.numMaxSize = 1;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;

        mExt_pm6_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mExt_pm6_mbox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "stem/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStem_concat_layer.get() == nullptr);
        mStem_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStem_concat_layer.get();
    }

    else if (!strcmp(layerName, "stage1_1/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage1_1_concat_layer.get() == nullptr);
        mStage1_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage1_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage1_2/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage1_2_concat_layer.get() == nullptr);
        mStage1_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage1_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage1_3/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage1_3_concat_layer.get() == nullptr);
        mStage1_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage1_3_concat_layer.get();
    }

    else if (!strcmp(layerName, "stage2_1/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage2_1_concat_layer.get() == nullptr);
        mStage2_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage2_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage2_2/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage2_2_concat_layer.get() == nullptr);
        mStage2_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage2_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage2_3/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage2_3_concat_layer.get() == nullptr);
        mStage2_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage2_3_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage2_4/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage2_4_concat_layer.get() == nullptr);
        mStage2_4_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage2_4_concat_layer.get();
    }

      else if (!strcmp(layerName, "stage3_1/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_1_concat_layer.get() == nullptr);
        mStage3_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage3_2/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_2_concat_layer.get() == nullptr);
        mStage3_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage3_3/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_3_concat_layer.get() == nullptr);
        mStage3_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_3_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_4/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_4_concat_layer.get() == nullptr);
        mStage3_4_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_4_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_5/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_5_concat_layer.get() == nullptr);
        mStage3_5_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_5_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_6/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_6_concat_layer.get() == nullptr);
        mStage3_6_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_6_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_7/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_7_concat_layer.get() == nullptr);
        mStage3_7_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_7_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_8/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage3_8_concat_layer.get() == nullptr);
        mStage3_8_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage3_8_concat_layer.get();
    }

      else if (!strcmp(layerName, "stage4_1/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage4_1_concat_layer.get() == nullptr);
        mStage4_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage4_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage4_2/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage4_2_concat_layer.get() == nullptr);
        mStage4_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage4_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage4_3/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage4_3_concat_layer.get() == nullptr);
        mStage4_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage4_3_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage4_4/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage4_4_concat_layer.get() == nullptr);
        mStage4_4_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage4_4_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage4_5/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage4_5_concat_layer.get() == nullptr);
        mStage4_5_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage4_5_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage4_6/concat"))
    {
        std::cout << layerName << std::endl;
        assert(mStage4_6_concat_layer.get() == nullptr);
        mStage4_6_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mStage4_6_concat_layer.get();
    }
     else if (!strcmp(layerName, "mbox_priorbox"))
    {
        std::cout << layerName << std::endl;
        assert(mBox_priorbox_layer.get() == nullptr);
        mBox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(2, true), nvPluginDeleter);
        return mBox_priorbox_layer.get();
    }


    else if (!strcmp(layerName, "mbox_loc"))
    {
        std::cout << layerName << std::endl;
        assert(mBox_loc_layer.get() == nullptr);
        mBox_loc_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mBox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf"))
    {
        std::cout << layerName << std::endl;
        assert(mBox_conf_layer.get() == nullptr);
        mBox_conf_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mBox_conf_layer.get();
    }

        //flatten
    else if (!strcmp(layerName, "ext/pm1_mbox_loc_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm1_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm1_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm1_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm1_mbox_conf_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm1_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm1_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm1_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm2_mbox_loc_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm2_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm2_mbox_conf_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm2_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm2_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm3_mbox_loc_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm3_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm3_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm3_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm3_mbox_conf_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm3_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm3_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm3_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm4_mbox_loc_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm4_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm4_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm4_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm4_mbox_conf_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm4_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm4_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm4_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm5_mbox_loc_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm5_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm5_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm5_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm5_mbox_conf_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm5_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm5_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm5_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm6_mbox_loc_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm6_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm6_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm6_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm6_mbox_conf_flat"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm6_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm6_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mExt_pm6_mbox_conf_flat_layer.get();
    }
   
    else if (!strcmp(layerName, "mbox_conf_flatten"))
    {
        std::cout << layerName << std::endl;
        assert(mMbox_conf_flat_layer.get() == nullptr);
        mMbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mMbox_conf_flat_layer.get();
    }


    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {
        std::cout << layerName << std::endl;
        assert(mMbox_conf_reshape.get() == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mMbox_conf_reshape = std::unique_ptr<Reshape<11>>(new Reshape<11>());
        return mMbox_conf_reshape.get();
    }
    //softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax"))
    {
        std::cout << layerName << std::endl;
        assert( mPluginSoftmax == nullptr);
        assert( nbWeights == 0 && weights == nullptr);
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin());
        return mPluginSoftmax.get();
    }
    else if (!strcmp(layerName, "detection_out"))
    {
        std::cout << layerName << std::endl;
        assert(mDetection_out.get() == nullptr);
        //tensor rt 3.0 
        //mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDDetectionOutputPlugin({true, false, 0, 21, 400, 200, 0.5, 0.45, CodeType_t::CENTER_SIZE}), nvPluginDeleter);
        //tensor rt 5



        DetectionOutputParameters params;
        params.backgroundLabelId = 0;
        params.codeType = CodeTypeSSD::CENTER_SIZE;
        params.keepTopK = 200;
        params.shareLocation = true;
        params.varianceEncodedInTarget = false;
        params.topK = 400;
        params.nmsThreshold = 0.4499;
        params.numClasses = 11;
        params.inputOrder[0] = 0;
        params.inputOrder[1] = 1;
        params.inputOrder[2] = 2;
        params.confidenceThreshold = 0.3;
        params.confSigmoid = false;
        params.isNormalized = true;



        mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDDetectionOutputPlugin(params), nvPluginDeleter);
        return mDetection_out.get();
    }
    else
    {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
       if (!strcmp(layerName, "ext/pm1_mbox_loc_perm"))
    {
        std::cout << layerName << std::endl;
        assert(mExt_pm1_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm1_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm1_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm1_mbox_conf_perm"))
    {
        assert(mExt_pm1_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm1_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm1_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm2_mbox_loc_perm"))
    {
        assert(mExt_pm2_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm2_mbox_conf_perm"))
    {
        assert(mExt_pm2_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm2_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm3_mbox_loc_perm"))
    {
        assert(mExt_pm3_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm3_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm3_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm3_mbox_conf_perm"))
    {
        assert(mExt_pm3_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm3_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm3_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm4_mbox_loc_perm"))
    {
        assert(mExt_pm4_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm4_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm4_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm4_mbox_conf_perm"))
    {
        assert(mExt_pm4_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm4_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm4_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm5_mbox_loc_perm"))
    {
        assert(mExt_pm5_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm5_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm5_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm5_mbox_conf_perm"))
    {
        assert(mExt_pm5_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm5_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm5_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm6_mbox_loc_perm"))
    {
        assert(mExt_pm6_mbox_loc_perm_layer.get() == nullptr);
        mExt_pm6_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm6_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm6_mbox_conf_perm"))
    {
        assert(mExt_pm6_mbox_conf_perm_layer.get() == nullptr);
        mExt_pm6_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm6_mbox_conf_perm_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm1_mbox_priorbox"))
    {
        assert(mExt_pm1_mbox_priorbox_layer.get() == nullptr);
        float min_size = 30.3999996185, max_size = 60.7999992371, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        mExt_pm1_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm1_mbox_priorbox_layer.get();
    }
      else if (!strcmp(layerName, "ext/pm2_mbox_priorbox"))
    {
        assert(mExt_pm2_mbox_priorbox_layer.get() == nullptr);
        float min_size = 60.7999992371, max_size = 112.480003357, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        mExt_pm2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm2_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm3_mbox_priorbox"))
    {
        assert(mExt_pm3_mbox_priorbox_layer.get() == nullptr);
        float min_size = 112.480003357, max_size = 164.160003662, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        mExt_pm3_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm3_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm4_mbox_priorbox"))
    {
        assert(mExt_pm4_mbox_priorbox_layer.get() == nullptr);
        float min_size = 164.160003662, max_size = 215.839996338, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        mExt_pm4_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm4_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm5_mbox_priorbox"))
    {
        assert(mExt_pm5_mbox_priorbox_layer.get() == nullptr);
        float min_size = 215.839996338, max_size = 267.519989014, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        mExt_pm5_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm5_mbox_priorbox_layer.get();
    }

      else if (!strcmp(layerName, "ext/pm6_mbox_priorbox"))
    {
        assert(mExt_pm6_mbox_priorbox_layer.get() == nullptr);
        float min_size = 267.519989014, max_size = 319.200012207, aspect_ratio[3] = {1.0, 2.0, 3.0}; //aspect_ratio[2] = {1.0, 2.0}; 
        mExt_pm6_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mExt_pm6_mbox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "stem/concat"))
    {
        assert(mStem_concat_layer.get() == nullptr);
        mStem_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStem_concat_layer.get();
    }

    else if (!strcmp(layerName, "stage1_1/concat"))
    {
        assert(mStage1_1_concat_layer.get() == nullptr);
        mStage1_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage1_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage1_2/concat"))
    {
        assert(mStage1_2_concat_layer.get() == nullptr);
        mStage1_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage1_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage1_3/concat"))
    {
        assert(mStage1_3_concat_layer.get() == nullptr);
        mStage1_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage1_3_concat_layer.get();
    }

    else if (!strcmp(layerName, "stage2_1/concat"))
    {
        assert(mStage2_1_concat_layer.get() == nullptr);
        mStage2_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage2_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage2_2/concat"))
    {
        assert(mStage2_2_concat_layer.get() == nullptr);
        mStage2_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage2_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage2_3/concat"))
    {
        assert(mStage2_3_concat_layer.get() == nullptr);
        mStage2_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage2_3_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage2_4/concat"))
    {
        assert(mStage2_4_concat_layer.get() == nullptr);
        mStage2_4_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage2_4_concat_layer.get();
    }

      else if (!strcmp(layerName, "stage3_1/concat"))
    {
        assert(mStage3_1_concat_layer.get() == nullptr);
        mStage3_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage3_2/concat"))
    {
        assert(mStage3_2_concat_layer.get() == nullptr);
        mStage3_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage3_3/concat"))
    {
        assert(mStage3_3_concat_layer.get() == nullptr);
        mStage3_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_3_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_4/concat"))
    {
        assert(mStage3_4_concat_layer.get() == nullptr);
        mStage3_4_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_4_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_5/concat"))
    {
        assert(mStage3_5_concat_layer.get() == nullptr);
        mStage3_5_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_5_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_6/concat"))
    {
        assert(mStage3_6_concat_layer.get() == nullptr);
        mStage3_6_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_6_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_7/concat"))
    {
        assert(mStage3_7_concat_layer.get() == nullptr);
        mStage3_7_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_7_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage3_8/concat"))
    {
        assert(mStage3_8_concat_layer.get() == nullptr);
        mStage3_8_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage3_8_concat_layer.get();
    }

      else if (!strcmp(layerName, "stage4_1/concat"))
    {
        assert(mStage4_1_concat_layer.get() == nullptr);
        mStage4_1_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage4_1_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage4_2/concat"))
    {
        assert(mStage4_2_concat_layer.get() == nullptr);
        mStage4_2_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage4_2_concat_layer.get();
    }
    else if (!strcmp(layerName, "stage4_3/concat"))
    {
        assert(mStage4_3_concat_layer.get() == nullptr);
        mStage4_3_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage4_3_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage4_4/concat"))
    {
        assert(mStage4_4_concat_layer.get() == nullptr);
        mStage4_4_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage4_4_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage4_5/concat"))
    {
        assert(mStage4_5_concat_layer.get() == nullptr);
        mStage4_5_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage4_5_concat_layer.get();
    }
     else if (!strcmp(layerName, "stage4_6/concat"))
    {
        assert(mStage4_6_concat_layer.get() == nullptr);
        mStage4_6_concat_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mStage4_6_concat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {
        assert(mBox_priorbox_layer.get() == nullptr);
        mBox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mBox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "mbox_loc"))
    {
        assert(mBox_loc_layer.get() == nullptr);
        mBox_loc_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mBox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf"))
    {
        assert(mBox_conf_layer.get() == nullptr);
        mBox_conf_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mBox_conf_layer.get();
    }

        //flatten
    else if (!strcmp(layerName, "ext/pm1_mbox_loc_flat"))
    {
        assert(mExt_pm1_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm1_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm1_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm1_mbox_conf_flat"))
    {
        assert(mExt_pm1_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm1_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm1_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm2_mbox_loc_flat"))
    {
        assert(mExt_pm2_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm2_mbox_conf_flat"))
    {
        assert(mExt_pm2_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm2_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm3_mbox_loc_flat"))
    {
        assert(mExt_pm3_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm3_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm3_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm3_mbox_conf_flat"))
    {
        assert(mExt_pm3_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm3_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm3_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm4_mbox_loc_flat"))
    {
        assert(mExt_pm4_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm4_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm4_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm4_mbox_conf_flat"))
    {
        assert(mExt_pm4_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm4_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm4_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm5_mbox_loc_flat"))
    {
        assert(mExt_pm5_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm5_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm5_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm5_mbox_conf_flat"))
    {
        assert(mExt_pm5_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm5_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm5_mbox_conf_flat_layer.get();
    }
     else if (!strcmp(layerName, "ext/pm6_mbox_loc_flat"))
    {
        assert(mExt_pm6_mbox_loc_flat_layer.get() == nullptr);
        mExt_pm6_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm6_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "ext/pm6_mbox_conf_flat"))
    {
        assert(mExt_pm6_mbox_conf_flat_layer.get() == nullptr);
        mExt_pm6_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mExt_pm6_mbox_conf_flat_layer.get();
    }
   
    else if (!strcmp(layerName, "mbox_conf_flatten"))
    {
        assert(mMbox_conf_flat_layer.get() == nullptr);
        mMbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mMbox_conf_flat_layer.get();
    }


    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {
        assert(mMbox_conf_reshape.get() == nullptr);
       // assert(nbWeights == 0 && weights == nullptr);
        mMbox_conf_reshape = std::unique_ptr<Reshape<11>>(new Reshape<11>(serialData, serialLength));
        return mMbox_conf_reshape.get();
    }
    //softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax"))
    {
        assert( mPluginSoftmax == nullptr);
       
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin(serialData, serialLength));
        return mPluginSoftmax.get();
    }
    else if (!strcmp(layerName, "detection_out"))
    {
        assert(mDetection_out.get() == nullptr);
        //tensor rt 3.0 
        //mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDDetectionOutputPlugin({true, false, 0, 21, 400, 200, 0.5, 0.45, CodeType_t::CENTER_SIZE}), nvPluginDeleter);
        //tensor rt 5
       

        mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
        return mDetection_out.get();
    }
    else
    {
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char* name)
{
    return (!strcmp(name, "ext/pm1_mbox_loc_perm")
            || !strcmp(name, "ext/pm1_mbox_conf_perm")
            || !strcmp(name, "ext/pm2_mbox_loc_perm")
            || !strcmp(name, "ext/pm2_mbox_conf_perm") 
            || !strcmp(name, "ext/pm3_mbox_loc_perm")
            || !strcmp(name, "ext/pm3_mbox_conf_perm") 
            || !strcmp(name, "ext/pm4_mbox_loc_perm")
            || !strcmp(name, "ext/pm4_mbox_conf_perm") 
            || !strcmp(name, "ext/pm5_mbox_loc_perm")
            || !strcmp(name, "ext/pm5_mbox_conf_perm") 
            || !strcmp(name, "ext/pm6_mbox_loc_perm")
            || !strcmp(name, "ext/pm6_mbox_conf_perm") 
            || !strcmp(name, "ext/pm1_mbox_priorbox") 
            || !strcmp(name, "ext/pm2_mbox_priorbox") 
            || !strcmp(name, "ext/pm3_mbox_priorbox") 
            || !strcmp(name, "ext/pm4_mbox_priorbox") 
            || !strcmp(name, "ext/pm5_mbox_priorbox") 
            || !strcmp(name, "ext/pm6_mbox_priorbox") 
            || !strcmp(name, "stem/concat") 
            || !strcmp(name, "stage1_1/concat") 
            || !strcmp(name, "stage1_2/concat") 
            || !strcmp(name, "stage1_3/concat") 
            || !strcmp(name, "stage2_1/concat") 
            || !strcmp(name, "stage2_2/concat") 
            || !strcmp(name, "stage2_3/concat") 
            || !strcmp(name, "stage2_4/concat") 
            || !strcmp(name, "stage3_1/concat") 
            || !strcmp(name, "stage3_2/concat") 
            || !strcmp(name, "stage3_3/concat") 
            || !strcmp(name, "stage3_4/concat") 
            || !strcmp(name, "stage3_5/concat") 
            || !strcmp(name, "stage3_6/concat") 
            || !strcmp(name, "stage3_7/concat") 
            || !strcmp(name, "stage3_8/concat") 
            || !strcmp(name, "stage4_1/concat") 
            || !strcmp(name, "stage4_2/concat") 
            || !strcmp(name, "stage4_3/concat") 
            || !strcmp(name, "stage4_4/concat") 
            || !strcmp(name, "stage4_5/concat") 
            || !strcmp(name, "stage4_6/concat") 
            || !strcmp(name, "mbox_loc")
            || !strcmp(name, "mbox_conf")
            || !strcmp(name, "ext/pm1_mbox_loc_flat")
            || !strcmp(name, "ext/pm1_mbox_conf_flat")
            || !strcmp(name, "ext/pm2_mbox_loc_flat")
            || !strcmp(name, "ext/pm2_mbox_conf_flat")
            || !strcmp(name, "ext/pm3_mbox_loc_flat")
            || !strcmp(name, "ext/pm3_mbox_conf_flat")
            || !strcmp(name, "ext/pm4_mbox_loc_flat")
            || !strcmp(name, "ext/pm4_mbox_conf_flat")
            || !strcmp(name, "ext/pm5_mbox_loc_flat")
            || !strcmp(name, "ext/pm5_mbox_conf_flat")
            || !strcmp(name, "ext/pm6_mbox_loc_flat")
            || !strcmp(name, "ext/pm6_mbox_conf_flat")
            || !strcmp(name, "mbox_conf_reshape")
            || !strcmp(name, "mbox_conf_flatten")
            || !strcmp(name, "mbox_loc")
            || !strcmp(name, "mbox_conf")
            || !strcmp(name, "mbox_priorbox")
            || !strcmp(name, "detection_out")
            || !strcmp(name, "mbox_conf_softmax"));


}

void PluginFactory::destroyPlugin()
{
   

    mExt_pm1_mbox_loc_perm_layer.release();
    mExt_pm1_mbox_conf_perm_layer.release();
    mExt_pm2_mbox_loc_perm_layer.release();
    mExt_pm2_mbox_conf_perm_layer.release();
    mExt_pm3_mbox_loc_perm_layer.release();
    mExt_pm3_mbox_conf_perm_layer.release();
    mExt_pm4_mbox_loc_perm_layer.release();
    mExt_pm4_mbox_conf_perm_layer.release();
    mExt_pm5_mbox_loc_perm_layer.release();
    mExt_pm5_mbox_conf_perm_layer.release();
    mExt_pm6_mbox_loc_perm_layer.release();
    mExt_pm6_mbox_conf_perm_layer.release();

    mExt_pm1_mbox_priorbox_layer.release();
    mExt_pm2_mbox_priorbox_layer.release();
    mExt_pm3_mbox_priorbox_layer.release();
    mExt_pm4_mbox_priorbox_layer.release();
    mExt_pm5_mbox_priorbox_layer.release();
    mExt_pm6_mbox_priorbox_layer.release();

    mStem_concat_layer.release();
    mStage1_1_concat_layer.release(); 
    mStage1_2_concat_layer.release();
    mStage1_3_concat_layer.release();

    mStage2_1_concat_layer.release(); 
    mStage2_2_concat_layer.release();
    mStage2_3_concat_layer.release();
    mStage2_4_concat_layer.release();


    mStage3_1_concat_layer.release();
    mStage3_2_concat_layer.release();
    mStage3_3_concat_layer.release();
    mStage3_4_concat_layer.release();
    mStage3_5_concat_layer.release(); 
    mStage3_6_concat_layer.release();
    mStage3_7_concat_layer.release();
    mStage3_8_concat_layer.release();


    mStage4_1_concat_layer.release(); 
    mStage4_2_concat_layer.release();
    mStage4_3_concat_layer.release();
    mStage4_4_concat_layer.release();
    mStage4_5_concat_layer.release();
    mStage4_6_concat_layer.release();


    mExt_pm1_mbox_loc_perm_layer= nullptr;
    mExt_pm1_mbox_conf_perm_layer= nullptr;
    mExt_pm2_mbox_loc_perm_layer= nullptr;
    mExt_pm2_mbox_conf_perm_layer= nullptr;
    mExt_pm3_mbox_loc_perm_layer= nullptr;
    mExt_pm3_mbox_conf_perm_layer = nullptr;
    mExt_pm4_mbox_loc_perm_layer= nullptr;
    mExt_pm4_mbox_conf_perm_layer= nullptr;
    mExt_pm5_mbox_loc_perm_layer= nullptr;
    mExt_pm5_mbox_conf_perm_layer= nullptr;
    mExt_pm6_mbox_loc_perm_layer= nullptr;
    mExt_pm6_mbox_conf_perm_layer= nullptr;

    mExt_pm1_mbox_priorbox_layer= nullptr;
    mExt_pm2_mbox_priorbox_layer= nullptr;
    mExt_pm3_mbox_priorbox_layer= nullptr;
    mExt_pm4_mbox_priorbox_layer= nullptr;
    mExt_pm5_mbox_priorbox_layer= nullptr;
    mExt_pm6_mbox_priorbox_layer= nullptr;

    mStem_concat_layer= nullptr;
    mStage1_1_concat_layer = nullptr;
    mStage1_2_concat_layer= nullptr;
    mStage1_3_concat_layer= nullptr;

    mStage2_1_concat_layer = nullptr;
    mStage2_2_concat_layer= nullptr;
    mStage2_3_concat_layer= nullptr;
    mStage2_4_concat_layer= nullptr;


    mStage3_1_concat_layer = nullptr;
    mStage3_2_concat_layer= nullptr;
    mStage3_3_concat_layer= nullptr;
    mStage3_4_concat_layer= nullptr;
    mStage3_5_concat_layer = nullptr;
    mStage3_6_concat_layer= nullptr;
    mStage3_7_concat_layer= nullptr;
    mStage3_8_concat_layer= nullptr;


    mStage4_1_concat_layer = nullptr;
    mStage4_2_concat_layer= nullptr;
    mStage4_3_concat_layer= nullptr;
    mStage4_4_concat_layer= nullptr;
    mStage4_5_concat_layer = nullptr;
    mStage4_6_concat_layer= nullptr;
    
    mBox_priorbox_layer.release();
    mBox_priorbox_layer = nullptr;
    mBox_loc_layer.release();
    mBox_loc_layer = nullptr;
    mBox_conf_layer.release();
    mBox_conf_layer = nullptr;

    mExt_pm1_mbox_loc_flat_layer.release();
    mExt_pm1_mbox_conf_flat_layer.release();
    mExt_pm2_mbox_loc_flat_layer.release();
    mExt_pm2_mbox_conf_flat_layer.release();
    mExt_pm3_mbox_loc_flat_layer.release();
    mExt_pm3_mbox_conf_flat_layer.release();
    mExt_pm4_mbox_loc_flat_layer.release();
    mExt_pm4_mbox_conf_flat_layer.release();
    mExt_pm5_mbox_loc_flat_layer.release();
    mExt_pm5_mbox_conf_flat_layer.release();
    mExt_pm6_mbox_loc_flat_layer.release();
    mExt_pm6_mbox_conf_flat_layer.release();

    mExt_pm1_mbox_loc_flat_layer= nullptr;
    mExt_pm1_mbox_conf_flat_layer= nullptr;
    mExt_pm2_mbox_loc_flat_layer= nullptr;
    mExt_pm2_mbox_conf_flat_layer= nullptr;
    mExt_pm3_mbox_loc_flat_layer= nullptr;
    mExt_pm3_mbox_conf_flat_layer= nullptr;
    mExt_pm4_mbox_loc_flat_layer= nullptr;
    mExt_pm4_mbox_conf_flat_layer= nullptr;
    mExt_pm5_mbox_loc_flat_layer= nullptr;
    mExt_pm5_mbox_conf_flat_layer= nullptr;
    mExt_pm6_mbox_loc_flat_layer= nullptr;
    mExt_pm6_mbox_conf_flat_layer= nullptr;

    mMbox_conf_flat_layer.release();
    mMbox_conf_flat_layer = nullptr;
    mMbox_conf_reshape.release();
    mMbox_conf_reshape = nullptr;
    mPluginSoftmax.release();
    mPluginSoftmax = nullptr;
    mDetection_out.release();
    mDetection_out = nullptr;
}
