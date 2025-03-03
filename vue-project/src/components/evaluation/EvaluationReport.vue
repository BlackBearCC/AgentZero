<template>
  <div class="report-container">
    <div class="report-header">
      <h2 class="report-title">评估报告</h2>
      <div class="report-actions">
        <button @click="exportReportCSV" class="crt-button export-btn">
          <span class="button-text">[ 导出报告(CSV) ]</span>
          <div class="button-icon">📊</div>
        </button>
      </div>
    </div>
    <div class="score-overview">
      <div class="score-card">
        <div class="score-value">{{ evaluationStats.overall_scores.final_score }}</div>
        <div class="score-label">总体评分</div>
      </div>
      <div class="score-card">
        <div class="score-value">{{ evaluationStats.overall_scores.role_score }}</div>
        <div class="score-label">角色评分</div>
      </div>
      <div class="score-card">
        <div class="score-value">{{ evaluationStats.overall_scores.dialogue_score }}</div>
        <div class="score-label">对话评分</div>
      </div>
    </div>
    <div class="report-details">
      <h3>详细信息</h3>
      <pre>{{ evaluationStats | json }}</pre>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    evaluationStats: {
      type: Object,
      required: true,
    },
  },
  methods: {
    exportReportCSV() {
      // 导出报告逻辑
      const csvContent = this.generateCSV();
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `评估报告_${new Date().toISOString().slice(0, 10)}.csv`;
      link.click();
    },
    generateCSV() {
      const headers = ['评估项目', '分数'];
      const rows = [
        ['总体评分', this.evaluationStats.overall_scores.final_score],
        ['角色评分', this.evaluationStats.overall_scores.role_score],
        ['对话评分', this.evaluationStats.overall_scores.dialogue_score],
      ];
      return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    },
  },
};
</script>

<style scoped>
.report-container {
  background: rgba(30, 30, 40, 0.8);
  padding: 1rem;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.score-overview {
  display: flex;
  justify-content: space-around;
  margin: 1rem 0;
}

.score-card {
  text-align: center;
}

.report-details {
  margin-top: 1rem;
}
</style> 