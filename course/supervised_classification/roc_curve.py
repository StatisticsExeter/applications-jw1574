import plotly.graph_objects as go


def _plot_roc_curve(lda_roc, qda_roc):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lda_roc['fpr'], y=lda_roc['tpr'],
        mode='lines',
        name=f'ROC curve from LDA (AUC = {lda_roc["roc_auc"]:.2f})'
    ))

    fig.add_trace(go.Scatter(
        x=qda_roc['fpr'], y=qda_roc['tpr'],
        mode='lines',
        name=f'ROC curve from QDA (AUC = {qda_roc["roc_auc"]:.2f})'
    ))

    # Test expects a random diagonal trace as the 3rd trace
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )

    return fig
