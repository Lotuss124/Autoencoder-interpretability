# 加载必要的包
library(ggplot2)
library(ggsci)
library(ggsignif)

# 示例数据框
coef_df <- data.frame(
  Variable = c("Intercept", "C(dropout_factor)[T.0.2]", "C(dropout_factor)[T.0.4]",
               "C(dropout_factor)[T.0.6]", "C(dropout_factor)[T.0.8]", "bn",
               "C(dropout_factor)[T.0.2]:bn", "C(dropout_factor)[T.0.4]:bn",
               "C(dropout_factor)[T.0.6]:bn", "C(dropout_factor)[T.0.8]:bn"),
  Coef = c(0.417, 0.244, 0.235, 0.027, -0.390, -0.015, 0.065, 0.218, 0.462, 0.908),
  Lower_CI = c(0.384, 0.197, 0.188, -0.020, -0.436, -0.062, -0.001, 0.152, 0.396, 0.842),
  Upper_CI = c(0.450, 0.290, 0.282, 0.074, -0.343, 0.031, 0.131, 0.284, 0.528, 0.974),
  p_value = c(0.000, 0.000, 0.000, 0.260, 0.000, 0.515, 0.053, 0.000, 0.000, 0.000)
)

# 添加变量显示名称
coef_df$Variable_Display <- c("Intercept", "Dropout 0.2", "Dropout 0.4",
                              "Dropout 0.6", "Dropout 0.8", "BN Main Effect",
                              "Dropout 0.2 × BN", "Dropout 0.4 × BN",
                              "Dropout 0.6 × BN", "Dropout 0.8 × BN")

# 计算相对于基准水平的绝对效应
baseline_coef <- coef_df$Coef[1]
baseline_lower <- coef_df$Lower_CI[1]
baseline_upper <- coef_df$Upper_CI[1]

absolute_df <- data.frame(
  Variable = coef_df$Variable,
  Variable_Display = coef_df$Variable_Display,
  Coef = coef_df$Coef + baseline_coef,
  Lower_CI = coef_df$Lower_CI + baseline_lower,
  Upper_CI = coef_df$Upper_CI + baseline_upper,
  p_value = coef_df$p_value
)

# 过滤掉基准水平（Intercept）本身
absolute_df <- absolute_df[absolute_df$Variable != "Intercept", ]

# 添加分组信息（特别处理bn主效应）
absolute_df$bn_status <- ifelse(grepl("× BN", absolute_df$Variable_Display) | absolute_df$Variable == "bn", 
                                "With BN", "Without BN")

absolute_df$dropout_level <- gsub("Dropout | × BN|BN Main Effect", "", absolute_df$Variable_Display)
absolute_df$dropout_level[absolute_df$Variable == "bn"] <- "0.0"  # 单独处理bn主效应

# 创建排序变量：先按dropout水平，再按BN状态（Without BN在前）
absolute_df$sort_var <- as.numeric(factor(absolute_df$dropout_level, levels = c("0.0", "0.2", "0.4", "0.6", "0.8"))) * 10 + 
  ifelse(absolute_df$bn_status == "With BN", 1, 0)

# 显著性标记
absolute_df$sig_label <- ifelse(absolute_df$p_value < 0.001, "***",
                                ifelse(absolute_df$p_value < 0.01, "**",
                                       ifelse(absolute_df$p_value < 0.05, "*", "")))

# 创建自定义标签
absolute_df$display_label <- absolute_df$Variable_Display

# 改进版森林图 - 按dropout水平分组排序
p <- ggplot(absolute_df, aes(x = reorder(display_label, sort_var), 
                             y = Coef,
                             color = bn_status)) +
  # 基准线
  geom_hline(yintercept = baseline_coef, 
             linetype = "dashed", 
             color = "grey40",
             linewidth = 1) +
  
  # 置信区间
  geom_errorbar(aes(ymin = Lower_CI, 
                    ymax = Upper_CI),
                width = 0.2,
                linewidth = 1.2) +
  
  # 点估计
  geom_point(size = 3) +
  
  # 显著性标记
  geom_text(aes(label = sig_label, 
                y = Upper_CI + 0.05),
            color = "black",
            size = 5,
            vjust = 0.5) +
  
  # 添加分组分隔线
  geom_vline(xintercept = c(1.5, 3.5, 5.5, 7.5), 
             linetype = "dotted", 
             color = "grey70",
             alpha = 0.7) +
  
  # 坐标轴和主题
  coord_flip() +
  labs(x = "", 
       y = "Coefficient Value (Absolute Effect)",
       color = "") +
  
  # 美化设置
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_line(color = "grey90", linetype = "dotted"),
    axis.text = element_text(color = "black"),
    axis.text.y = element_text(hjust = 0.5),
    legend.position = c(0.85, 0.15),
    legend.background = element_blank(),
    legend.key = element_blank(),
    plot.margin = margin(1, 1, 1, 1, "cm")
  ) +
  
  # 颜色设置（确保bn主效应使用With BN的颜色）
  scale_color_manual(values = c("With BN" = "#F4A455", 
                                "Without BN" = "#7193B8"),
                     guide = guide_legend(override.aes = list(size = 3))) +
  scale_y_continuous(breaks = seq(-0.1, 1.5, by = 0.2),
                     limits = c(-0.1, 1.5),
                     expand = expansion(mult = c(0, 0.05))) 

# 显示图形
print(p)

# 保存图形
ggsave("dropout_bn_comparison_final.pdf", 
       plot = p,
       width = 9, 
       height = 6,
       units = "in",
       dpi = 600,
       bg = "white")